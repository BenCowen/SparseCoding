from lib.core.task import Task
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import SequentialSampler, DataLoader
import gc


def create_sequential_dataloader(original_loader):
    """Creates a new DataLoader with SequentialSampler to ensure consistent order."""
    dataset = original_loader.dataset
    batch_size = 16
    num_workers = original_loader.num_workers
    pin_memory = original_loader.pin_memory

    # Create a new DataLoader with SequentialSampler
    sequential_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),  # Use SequentialSampler for consistent order
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return sequential_loader


class wrapper:
    def __init__(self, loader, prep):
        self.loader = loader  # Original DataLoader
        self.prep = prep  # The encoder function/model

    def __iter__(self):
        self.iterator = iter(self.loader)  # Ensure the DataLoader order is maintained
        return self

    def __next__(self):
        # Get the next batch from the loader
        data, label = next(self.iterator)
        # Apply the encoding (prep) to the batch
        return self.prep(data), label


class TsneB4After(Task):
    def __init__(self):
        self.wrapped = None
        self.save_dir = None
        self.dataloader = None
        self.decoder = None
        self.encoder = None

    def run_task(self, stuff: dict) -> dict:
        # Load the encoder and decoder
        self.decoder = stuff['trained-decoder']
        self.encoder = stuff['trained-encoder']
        self.save_dir = stuff['save_dir']

        # Load the original validation DataLoader
        self.dataloader = create_sequential_dataloader(stuff['dataloaders'].valid_loader)

        # Ensure reproducibility and order by fixing the DataLoader sampler
        self._fix_loader_order()

        # t-SNE on the original data
        self._tsne_visualize(self.dataloader,
                             title="Original Data",
                             save_file=self.save_dir / "before_encoding.png")

        # Wrap the DataLoader with the encoder
        self.wrapped = wrapper(self.dataloader, self._encode_batch)

        # t-SNE on the encoded data
        self._tsne_visualize(self.wrapped,
                             title="Encoded Data",
                             save_file=self.save_dir / "after_encoding.png")

    def _encode_batch(self, batch):
        # Assumes the batch is a tensor or a tuple where the first element is the input data
        data, _ = batch  # Assuming batch contains (data, labels) or similar
        return self.encoder(data)  # Encode the input data

    def _fix_loader_order(self):
        # If using SubsetRandomSampler, change it to SequentialSampler for consistent order
        if isinstance(self.dataloader.sampler, torch.utils.data.SubsetRandomSampler):
            self.dataloader.sampler = torch.utils.data.SequentialSampler(self.dataloader.dataset)

    @staticmethod
    def _tsne_visualize(loader,
                        title="t-SNE Visualization",
                        save_file=None):
        data_list = []
        labels_list = []

        # Collect all data and labels from the DataLoader
        for batch in loader:
            data, labels = batch
            data_list.append(data.cpu())
            labels_list.append(labels.cpu())
            gc.collect()

        # Stack the collected batches into a single tensor
        data = torch.cat(data_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        data_tsne = tsne.fit_transform(data)

        # Plot the t-SNE result
        plt.figure(figsize=(8, 6))
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', s=5)
        plt.colorbar()
        plt.title(title)
        if save_file:
            plt.savefig(save_file)
            plt.close()
            print(f"Check image at {save_file}")
        else:
            plt.show(save_file)
