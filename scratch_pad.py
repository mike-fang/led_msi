import usb.core
import usb.util
"""
Vendor 0x403
Product 0x6001
"""
dev = usb.core.find(idVendor=0x403, idProduct=0x6001)
buf = [0]
print(dev)
dev.write(0x02, buf, 500)
dev.reset()
