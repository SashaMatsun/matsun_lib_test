from best_gan_lib import Model
from best_gan_lib.generators import DCGANGenerator
from best_gan_lib.discriminators import DCGANDiscriminator
from best_gan_lib.generators import CGANGenerator
from best_gan_lib.discriminators import CGANDiscriminator

g = CGANGenerator(64, 128, 3, 10)
d = CGANDiscriminator(64, 3, 10)

model = Model(g, d)
