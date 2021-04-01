import sys
sys.path.append('utils_dir')
import utils
sys.path.append('discr_dir')
from dcgan_d import DCGAN_Discriminator
sys.path.append('gen_dir')
from dcgan_g import DCGAN_Generator
from Model import Model


discr = DCGAN_Discriminator(64, 3)
gen = DCGAN_Generator(64, 128, 3)
model = Model(gen, discr)