""" Confidence metrics for evaluating segmentation output """
from evaluators.metrics import Simplicity, Convexity


class Confidence:
    def __init__(self, metric, cutoff=1.1):
        self.cutoff = cutoff
        self.metric = metric

    def __call__(self, outputs, vals=False):
        if len(outputs.shape) == 3:
            outputs = outputs.clone().unsqueeze(0)
        assert len(outputs.shape) == 4, "expected input of shape 1, C, H, W or C, H, W"
        res = self.metric.process_single(outputs, outputs)
        if res is None:
            return True  # failed to find simplicity
        if not vals:
            return res > self.cutoff
        else:
            return res


class ConvexityConfidence(Confidence):
    def __init__(self, label_val=1, cutoff=0.9):
        metric = Convexity(label_val)
        super(ConvexityConfidence, self).__init__(metric, cutoff)


class SimplicityConfidence(Confidence):
    def __init__(self, label_val=1, cutoff=0.7):
        metric = Simplicity(label_val)
        super(SimplicityConfidence, self).__init__(metric, cutoff)


class SimplicityConvexityConfidence:
    def __init__(self, label_val=1, simplicity_cutoff=0.7, convexity_cutoff=0.9):
        self.simplicity = SimplicityConfidence(label_val, simplicity_cutoff)
        self.convexity = ConvexityConfidence(label_val, convexity_cutoff)

    def __call__(self, output, vals=False):
        if vals:
            return dict(simplicity=self.simplicity(output, True), convexity=self.convexity(output, True))
        else:
            return self.simplicity(output, False) and self.convexity(output, False)
