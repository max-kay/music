from matplotlib import pyplot as plt
import colored_traceback
import instruments as inst

colored_traceback.add_hook()


bass = inst.Instrument.from_json('instruments/soft_bass.json')
fig = bass.plot()
plt.show()
