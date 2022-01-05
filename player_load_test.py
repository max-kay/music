import player
import instruments as inst

song = player.Player.from_json('players/test.json')

eff = inst.Effect({}, 'reverb')

song.add_effect(eff)

song.save_wav()
