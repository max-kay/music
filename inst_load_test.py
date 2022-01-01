from matplotlib import pyplot as plt
import instruments as inst

bass = inst.SynthBase.from_json('bass')
fig = bass.plot()
plt.show()