import chainer.functions as F
import chainer.links as L

from chainer import Chain

from networks.resnet.resnet import ResNet


class HandwritingNet(Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.feature_extractor = ResNet(18)
            self.classifier = L.Linear(None, 2)

    def forward(self, **kwargs):
        images = kwargs['image']
        features = self.feature_extractor(images)
        return self.classifier(features)
