import gc
import torch
from lib.tasks.trainer_abc import Trainer
import lib.UTILS.path_support as torch_aux
from lib.UTILS.visualizer import Visualizer


class SslTrainer(Trainer):
    def __init__(self, config):
        super(SslTrainer, self).__init__(config)
        self.viz = Visualizer(self.config['save-dir'])

    def train(self, model, dataset, loaded_objects={}):
        # Configure GPU usage with pinned memory dataloader
        model.add_to_device()
        # model = torch_aux.models_to_GPU([model])[0]

        # Assign loss function to self:
        loss_fcn = torch_aux.generate_loss_function(self.config['loss-config'], dataset)
        n_batches = min(self.max_batches, len(dataset))
        self.batches_per_print = n_batches // self.prints_per_epoch

        # Initialize the optimizer for our model
        self.optimizer, self.scheduler = torch_aux.setup_optimizer(self.config['optimizer-config'], model)

        # Keep record of the training process
        self.training_record = {'epoch': 0,
                                'loss-hist': []}

        self._parse_loaded_objects(loaded_objects)
        self.epoch = self.training_record['epoch']

        # Ready to train:
        while self.epoch < self.max_epoch:
            # Visualizations each epoch:
            # self.epoch_visualizations(model, dataset.valid_loader)

            # Add an epoch to the record:
            self.training_record['epoch'] = self.epoch
            self.training_record['loss-hist'].append([])

            # Train for 1 epoch:
            model.train(mode=True)
            for batch_idx, batch in enumerate(dataset.train_loader):
                self.optimizer.zero_grad()
                batch = model.add_to_device(batch)
                # Forward pass
                loss_value = loss_fcn(batch, model(batch))

                # Backward pass:
                loss_value.backward()
                self.optimizer.step()
                self.scheduler.step()
                gc.collect()

                # Logistics
                detached_loss = loss_value.detach().cpu().numpy()
                self.training_record['loss-hist'][-1].append(detached_loss)
                self.batch_print(detached_loss, batch_idx, n_batches)

            # Save trainer
            self.save_train_state(model, self.training_record,
                                  self.optimizer, self.scheduler)

            self.epoch += 1
            self.epoch_visualizations(model, dataset.valid_loader)

    def plot_pytorch_tensor(self, ax, tensor):
        if tensor.ndim==3:
            ax.imshow(torch.permute(tensor, (0, 1, 2)).squeeze().detach().cpu().numpy())
        else:
            ax.imshow(torch.permute(tensor, (0, 2, 3, 1)).squeeze().detach().cpu().numpy())
        return ax

    @torch.no_grad()
    def epoch_visualizations(self, model, valid_loader):
        model.train(mode=False)
        N = 2
        # Show original and its reconstruction

        images = torch.stack([valid_loader.dataset[n] for n in range(N)])
        images = torch.stack([images, model(model.add_to_device(images))['recon'].cpu()])
        self.viz.array_plot(images, save_name='recons', color=True)
        imsz = images.shape[-2]
        # Now draw some random samples from latent space...
        z = model.add_to_device(torch.randn(N ** 2, model.latent_dim))
        self.viz.array_plot(model.decode(z).view(N,N,3,imsz,imsz),
                            save_name='samples', color=True)

        self.viz.plot_loss(self.training_record, save_name='loss_history')
