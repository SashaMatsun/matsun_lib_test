from best_gan_lib import Model
from best_gan_lib.generators import DCGANGenerator
from best_gan_lib.discriminators import DCGANDiscriminator

g = DCGANGenerator(64, 128, 3)
d = DCGANDiscriminator(64, 3)

model = Model(g, d)
