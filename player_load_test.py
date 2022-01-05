import player
import instruments as inst

song = player.Player.from_json('players/torpedo.json')

eff = inst.Effect('reverb', {})

song.add_effect(eff)

print(song)

song.save_wav()
