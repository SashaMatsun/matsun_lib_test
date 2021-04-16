from torch_gan import Model
from torch_gan.generators import DCGANGenerator
from torch_gan.discriminators import DCGANDiscriminator
from torch_gan.generators import CGANGenerator
from torch_gan.discriminators import CGANDiscriminator
from torch_gan.generators import P2PGenerator
from torch_gan.discriminators import P2PDiscriminator
from torch_gan.discriminators import InfoGANDiscriminator
from torch_gan.discriminators import BEGANDiscriminator
from torch_gan.generators import SAGANGenerator
from torch_gan.discriminators import SAGANDiscriminator


g = SAGANGenerator(64, 128, 3)
d = SAGANDiscriminator(64, 3)

model = Model(g, d)
