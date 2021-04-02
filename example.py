from best_gan_lib import Model
from best_gan_lib.generators import DCGANGenerator
from best_gan_lib.discriminators import DCGANDiscriminator
from best_gan_lib.generators import CGANGenerator
from best_gan_lib.discriminators import CGANDiscriminator
from best_gan_lib.generators import P2PGenerator
from best_gan_lib.discriminators import P2PDiscriminator


g = P2PGenerator(64, 128, 3)
d = P2PDiscriminator(64, 3)

model = Model(g, d)
