import instruments as osc
import matplotlib.pyplot as plt

env = osc.Envelope((.1, .1, .8, .6))

plt.subplots()
plt.plot(env.get_envelope(2))
plt.show()