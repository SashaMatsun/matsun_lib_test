from best_gan_lib import Model
from best_gan_lib.generators import DCGANGenerator
from best_gan_lib.discriminators import DCGANDiscriminator
from best_gan_lib.generators import CGANGenerator
from best_gan_lib.discriminators import CGANDiscriminator
from best_gan_lib.generators import P2PGenerator
from best_gan_lib.discriminators import P2PDiscriminator
from best_gan_lib.discriminators import InfoGANDiscriminator
from best_gan_lib.discriminators import BEGANDiscriminator
from best_gan_lib.generators import SAGANGenerator
from best_gan_lib.discriminators import SAGANDiscriminator


g = SAGANGenerator(64, 128, 3)
d = SAGANDiscriminator(64, 3)

model = Model(g, d)
