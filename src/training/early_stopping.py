from typing import Tuple


class EarlyStopping:
    def __init__(self, early_stop_patience, min_delta = None, early_stop_min_delta_patience = None):
        self.inc_start_loss = 0.
        self.prev_metric_val = float("inf")
        self.inc_iter = 0
        self.min_delta_iter = 0
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta_patience = early_stop_min_delta_patience if early_stop_min_delta_patience is not None else float("inf")
        self.min_delta = min_delta if min_delta is not None else float("-inf")
        assert early_stop_patience >= 0
        assert (early_stop_min_delta_patience is None and min_delta is None) or (early_stop_min_delta_patience >= 0 and min_delta > 0)
    
    def add_metric(self, metric_val: float) -> Tuple[bool, bool]:
        """[summary]

        Args:
            val_loss (float): [description]

        Returns:
            Tuple[bool, bool]: (should_checkpoint, should_stop)
        """
        should_checkpoint = should_stop = False

        if self.inc_iter > self.early_stop_patience or self.min_delta_iter > self.early_stop_min_delta_patience:
            raise ValueError("Training not stopped")

        if metric_val > self.prev_metric_val:
            self.min_delta_iter = 0
            self.inc_iter += 1
            print('Early stop loss increased')
            if self.inc_iter == 1:
                print('Saving checkpoint due to increase of early stop loss')
                should_checkpoint = True
                self.inc_start_loss = self.prev_metric_val
            if self.inc_iter > self.early_stop_patience:
                should_stop = True
                print('Stopped due to increase of early stop loss')

        if self.inc_iter >= 1 and metric_val <= self.inc_start_loss:
            self.inc_iter = 0
            self.inc_start_loss = 0
            print('saving checkpoint due to decrease of early stop loss')
            should_checkpoint = True
        

        if metric_val <= self.prev_metric_val and self.prev_metric_val - metric_val < self.min_delta:
            self.min_delta_iter += 1
            print('Early stop loss delta < ', self.min_delta)
            if self.min_delta_iter > self.early_stop_min_delta_patience:
                should_stop = True
                should_checkpoint = True
        else:
            self.min_delta_iter = 0


        self.prev_metric_val = metric_val
        return (should_checkpoint, should_stop)
    
    @property
    def metric_increased(self) -> bool: 
        return self.inc_iter > 0