import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torchvision.transforms as transforms
from tqdm.auto import tqdm  # progress bar
from .feature_extractors import (
    ResNet18Embedding,
    VAEEmbedding,
    ClipEmbedding,
    KLVAEEmbedding,
)
import argparse

EPS_FACTOR = 1 / 255
ALPHA_FACTOR = 0.05
N_STEPS = 200
BATCH_SIZE = 4


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet18",
        choices=["resnet18", "clip", "klvae8", "sdxlvae", "klvae16"],
        help="Embedding backbone to use.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=2,
        choices=[2, 4, 6, 8],
        help="Attack strength multiplier (scales eps and alpha)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help=(
            "GPU index to run on (e.g. 0 for cuda:0). "
            "Use -1 to force CPU even if CUDA is available."
        ),
    )
    parsed_args = parser.parse_args()
    return parsed_args


def adv_emb_attack(
    wm_img_path: str,
    encoder: str,
    strength: float,
    output_path: str,
    device: torch.device,
):
    """Run the adversarial embedding attack with a tqdm progress bar.

    The output directory is automatically suffixed with the current strength value,
    e.g., ``output_path/strength_4`` for ``strength==4``.
    """

    # Validate paths
    if not os.path.isdir(wm_img_path):
        raise FileNotFoundError(
            f"Input path does not exist or is not a directory: {wm_img_path}"
        )

    # -------------------------------------------------------------
    # Create a *strength‑specific* directory to keep results tidy
    # -------------------------------------------------------------
    output_dir = os.path.join(output_path, f"strength_{strength}")
    os.makedirs(output_dir, exist_ok=True)

    # Select embedding model
    if encoder == "resnet18":
        embedding_model = ResNet18Embedding("last")
    elif encoder == "clip":
        embedding_model = ClipEmbedding()
    elif encoder == "klvae8":
        embedding_model = VAEEmbedding("stabilityai/sd-vae-ft-mse")
    elif encoder == "sdxlvae":
        embedding_model = VAEEmbedding("stabilityai/sdxl-vae")
    elif encoder == "klvae16":
        embedding_model = KLVAEEmbedding("kl-f16")
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")

    embedding_model = embedding_model.to(device).eval()
    print("Embedding model loaded on", device)

    # Data loader
    transform = transforms.ToTensor()
    wm_dataset = SimpleImageFolder(wm_img_path, transform=transform)
    wm_loader = DataLoader(
        wm_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    print("Data loaded! ->", len(wm_dataset), "images")

    # Attack configuration
    attack = WarmupPGDEmbedding(
        model=embedding_model,
        eps=EPS_FACTOR * strength,
        alpha=ALPHA_FACTOR * EPS_FACTOR * strength,
        steps=N_STEPS,
        device=device,
    )

    # Adversarial generation with progress bar
    for images, image_paths in tqdm(wm_loader, desc="Attacking", unit="batch"):
        images = images.to(device)
        images_adv = attack.forward(images)

        # Save results
        for img_adv, image_path in zip(images_adv, image_paths):
            save_path = os.path.join(output_dir, os.path.basename(image_path))
            save_image(img_adv, save_path)

    print("Attack finished! Adversarial images saved to", output_dir)


class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None, extensions=None):
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]
        self.root = root
        self.transform = transform
        self.extensions = extensions

        self.filenames = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and os.path.splitext(f)[1].lower() in self.extensions
        ]

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.filenames)


class WarmupPGDEmbedding:
    def __init__(
        self,
        model,
        device,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        loss_type="l2",
        random_start=True,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_type = loss_type
        self.random_start = random_start
        self.device = device

        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, images, init_delta=None):
        self.model.eval()
        images = images.clone().detach().to(self.device)

        original_embeddings = self.model(images).detach()

        # Initialize adversarial images
        if self.random_start:
            adv_images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()
        elif init_delta is not None:
            adv_images = torch.clamp(images + init_delta, 0, 1).detach()
        else:
            raise ValueError("init_delta must be provided when random_start is False")

        # PGD iterations
        for _ in range(self.steps):
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)
            cost = self.loss_fn(adv_embeddings, original_embeddings)

            grad = torch.autograd.grad(cost, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images


if __name__ == "__main__":
    args = parse_arguments()

    # ------------------------------------------------------------------
    # Determine the computation device based on --gpu flag and CUDA state
    # ------------------------------------------------------------------
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
        if args.gpu != -1:
            print("CUDA not available -- falling back to CPU")
    else:
        n_gpu = torch.cuda.device_count()
        if args.gpu >= n_gpu or args.gpu < -1:
            raise ValueError(
                f"Requested GPU index {args.gpu} is out of range (0‑{n_gpu-1})"
            )
        device = torch.device(f"cuda:{args.gpu}")

    adv_emb_attack(
        wm_img_path="data/wm_imgs",
        encoder=args.encoder,
        strength=args.strength,
        output_path="data/adv_out",
        device=device,
    )
