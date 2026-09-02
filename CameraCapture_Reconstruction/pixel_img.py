# Import the needed libraries
from matplotlib import pyplot as plt
import numpy as np
import struct
HEXADECIMAL_BYTES = [
0x99D7, 0x9CD7, 0x9CD7, 0xF9A5, 0x5495, 0x5195, 0x1395, 0xB196, 0x507E, 0x98CF, 0x99CF,
0x2953, 0xC749, 0xE849, 0x084B, 0x284B, 0xAA53, 0x8853, 0xC853, 0xE95B, 0x4B6C, 0x96F7,
// paste entire 25,344 RGB565 pixel hexadecimal value here
]
# Reformat the bytes into an image
raw_bytes = np.array(HEXADECIMAL_BYTES, dtype=np.uint16)
image = np.zeros((len(raw_bytes),3), dtype=int)
# Loop through all of the pixels and form the image
for i in range(len(raw_bytes)):
#Read 16-bit pixel
pixel = struct.unpack('>h', raw_bytes[i])[0]
#Convert RGB565 to RGB 24-bit
r = ((pixel >> 11) & 0x1f) << 3;
g = ((pixel >> 5) & 0x3f) << 2;
b = ((pixel >> 0) & 0x1f) << 3;
image[i] = [r,g,b]
image = np.reshape(image,(144, 176,3)) #QCIF resolution
# Show the image
plt.imshow(image)
plt.show()
