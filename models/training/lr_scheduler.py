"""
Learning rate schedulers for LSTM model training.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class CyclicLR(Callback):
    """
    Cyclical Learning Rate callback.
    
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    
    References:
    - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
    """
    
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000, mode='triangular',
                gamma=1.0, scale_fn=None, scale_mode='cycle'):
        """
        Initialize cyclical learning rate callback.
        
        Parameters:
        -----------
        base_lr : float, optional (default=0.001)
            Initial learning rate which is the lower boundary in the cycle.
        max_lr : float, optional (default=0.006)
            Upper boundary in the cycle. Functionally, it defines the cycle amplitude.
        step_size : int, optional (default=2000)
            Number of training iterations per half cycle.
        mode : str, optional (default='triangular')
            One of {triangular, triangular2, exp_range}.
        gamma : float, optional (default=1.0)
            Constant in 'exp_range' scaling function: gamma**(cycle iterations).
        scale_fn : function, optional (default=None)
            Custom scaling function: if None, auto select based on mode.
        scale_mode : str, optional (default='cycle')
            One of {cycle, iterations}.
        """
        super(CyclicLR, self).__init__()
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        
        self.clr_iterations = 0
        self.trn_iterations = 0
        self.history = {}
        
        # Initialize learning rate
        self._reset()
    
    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """
        Reset cyclical learning rate state.
        
        Parameters:
        -----------
        new_base_lr : float, optional (default=None)
            New base learning rate.
        new_max_lr : float, optional (default=None)
            New max learning rate.
        new_step_size : int, optional (default=None)
            New step size.
        """
        if new_base_lr:
            self.base_lr = new_base_lr
        if new_max_lr:
            self.max_lr = new_max_lr
        if new_step_size:
            self.step_size = new_step_size
        
        self.clr_iterations = 0
    
    def clr(self):
        """
        Calculate current learning rate.
        
        Returns:
        --------
        lr : float
            Current learning rate.
        """
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
    
    def on_train_begin(self, logs=None):
        """
        Initialize learning rate at the start of training.
        
        Parameters:
        -----------
        logs : dict, optional (default=None)
            Training logs.
        """
        if self.clr_iterations == 0:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            
            # Get initial learning rate
            self.base_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            self.history['lr'] = []
    
    def on_train_batch_end(self, batch, logs=None):
        """
        Update learning rate at the end of each training batch.
        
        Parameters:
        -----------
        batch : int
            Current batch index.
        logs : dict, optional (default=None)
            Training logs.
        """
        logs = logs or {}
        
        # Update iterations
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        # Calculate and set new learning rate
        lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        # Log learning rate
        self.history.setdefault('lr', []).append(lr)
        
        # Log other metrics
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class StepDecay:
    """
    Step-based learning rate decay scheduler.
    
    Reduces the learning rate by a factor after a certain number of epochs.
    """
    
    def __init__(self, initial_lr=0.01, factor=0.5, drop_every=10):
        """
        Initialize step decay scheduler.
        
        Parameters:
        -----------
        initial_lr : float, optional (default=0.01)
            Initial learning rate.
        factor : float, optional (default=0.5)
            Factor to reduce learning rate by.
        drop_every : int, optional (default=10)
            Number of epochs between learning rate drops.
        """
        self.initial_lr = initial_lr
        self.factor = factor
        self.drop_every = drop_every
    
    def __call__(self, epoch):
        """
        Calculate learning rate for the current epoch.
        
        Parameters:
        -----------
        epoch : int
            Current epoch.
            
        Returns:
        --------
        lr : float
            Learning rate for the current epoch.
        """
        # Calculate learning rate drop
        exp = np.floor((1 + epoch) / self.drop_every)
        lr = self.initial_lr * (self.factor ** exp)
        
        return float(lr)


class PolynomialDecay:
    """
    Polynomial learning rate decay scheduler.
    
    Reduces the learning rate according to a polynomial function.
    """
    
    def __init__(self, initial_lr=0.01, final_lr=0.0001, decay_steps=100, power=1.0):
        """
        Initialize polynomial decay scheduler.
        
        Parameters:
        -----------
        initial_lr : float, optional (default=0.01)
            Initial learning rate.
        final_lr : float, optional (default=0.0001)
            Final learning rate.
        decay_steps : int, optional (default=100)
            Number of epochs to decay learning rate over.
        power : float, optional (default=1.0)
            Power of polynomial. 1.0 is linear decay.
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_steps = decay_steps
        self.power = power
    
    def __call__(self, epoch):
        """
        Calculate learning rate for the current epoch.
        
        Parameters:
        -----------
        epoch : int
            Current epoch.
            
        Returns:
        --------
        lr : float
            Learning rate for the current epoch.
        """
        # Calculate decay factor
        decay = (1 - epoch / self.decay_steps) ** self.power
        
        # Ensure decay is within bounds
        decay = max(0, min(1, decay))
        
        # Calculate new learning rate
        lr = self.final_lr + (self.initial_lr - self.final_lr) * decay
        
        return float(lr)


# Create a learning rate scheduler as a Keras callback
def create_lr_scheduler(schedule_type='step', **kwargs):
    """
    Create a learning rate scheduler callback.
    
    Parameters:
    -----------
    schedule_type : str, optional (default='step')
        Type of scheduler to create. One of {step, polynomial, cyclic}.
    **kwargs : dict
        Additional arguments for the scheduler.
        
    Returns:
    --------
    scheduler : Callback
        Learning rate scheduler callback.
    """
    if schedule_type == 'step':
        # Extract parameters for step decay
        initial_lr = kwargs.get('initial_lr', 0.01)
        factor = kwargs.get('factor', 0.5)
        drop_every = kwargs.get('drop_every', 10)
        
        # Create scheduler
        schedule = StepDecay(initial_lr, factor, drop_every)
        return tf.keras.callbacks.LearningRateScheduler(schedule)
    
    elif schedule_type == 'polynomial':
        # Extract parameters for polynomial decay
        initial_lr = kwargs.get('initial_lr', 0.01)
        final_lr = kwargs.get('final_lr', 0.0001)
        decay_steps = kwargs.get('decay_steps', 100)
        power = kwargs.get('power', 1.0)
        
        # Create scheduler
        schedule = PolynomialDecay(initial_lr, final_lr, decay_steps, power)
        return tf.keras.callbacks.LearningRateScheduler(schedule)
    
    elif schedule_type == 'cyclic':
        # Extract parameters for cyclic learning rate
        base_lr = kwargs.get('base_lr', 0.001)
        max_lr = kwargs.get('max_lr', 0.006)
        step_size = kwargs.get('step_size', 2000)
        mode = kwargs.get('mode', 'triangular')
        
        # Create scheduler
        return CyclicLR(base_lr, max_lr, step_size, mode)
    
    else:
        raise ValueError(f"Unknown scheduler type: {schedule_type}")


# Create a warmup learning rate scheduler
class WarmupLearningRateScheduler(Callback):
    """
    Learning rate scheduler with warmup.
    
    Gradually increases learning rate from 0 to initial_lr during warmup_epochs,
    then follows the specified schedule.
    """
    
    def __init__(self, schedule, warmup_epochs=5, initial_lr=0.01):
        """
        Initialize warmup scheduler.
        
        Parameters:
        -----------
        schedule : function
            Function that takes epoch as input and returns learning rate.
        warmup_epochs : int, optional (default=5)
            Number of epochs for warmup.
        initial_lr : float, optional (default=0.01)
            Initial learning rate after warmup.
        """
        super(WarmupLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        """
        Set learning rate at the beginning of each epoch.
        
        Parameters:
        -----------
        epoch : int
            Current epoch.
        logs : dict, optional (default=None)
            Training logs.
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
        else:
            # Follow schedule after warmup
            lr = self.schedule(epoch - self.warmup_epochs)
        
        # Set learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        # Log learning rate
        print(f"\nEpoch {epoch+1}: learning rate = {lr:.6f}") 