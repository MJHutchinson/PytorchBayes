from torch.nn import Module


class MFVIModule(Module):

    def forward(self, *input):
        raise NotImplementedError

    def _kl(self):
        return 0

    def kl(self):
        kl = self._kl()

        for child in self.children():
            if(isinstance(child, MFVIModule)):
                kl += child.kl()
            else:
                for child2 in child.children():
                    if (isinstance(child2, MFVIModule)):
                        kl += child2.kl()

        return kl