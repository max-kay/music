from music import player
import music.instruments as inst

song = player.Player.from_json('players/seven8.json')

# eff = inst.Effect('reverb', {'wet': 1, 'dry': .5})
# song.add_effect(eff)

song.save_json()
print(song)


song.save_wav()
