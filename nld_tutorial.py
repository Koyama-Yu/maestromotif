import nle.dataset as nld

# 1. Get the paths for your unzipped datasets
#path_to_nld_aa_taster = "./nld-aa-taster/nle_data"
path_to_nld_aa_taster = "./train_dir/skill_policy/debug2/ttyrecs"

# 2. Chose a database name/path. By default, most methods with use nld.db.DB (='ttyrecs.db')
#dbfilename = "ttyrecs.db"
dbfilename = "ttyrecs_test_debug5.db"

if not nld.db.exists(dbfilename):
    # 3. Create the db and add the directory
    nld.db.create(dbfilename)
    nld.add_nledata_directory(path_to_nld_aa_taster, "taster-dataset", dbfilename)


# NB: To add the NLE-AA data, or any data generated from nle, use `add_nledata_directory`.
# nld.add_nledata_directory(path_to_nld_aa, "nld-aa", dbfilename)

# NB: To add the NLE-NAO data, use the `add_altorg_directory`.
# nld.add_altorg_directory(path_to_nld_nao, "nld-nao", dbfilename)

# Create a connection to specify the database to use
db_conn = nld.db.connect(filename=dbfilename)

# Then you can inspect the number of games in each dataset:
print(f"NLD AA \"Taster\" Dataset has {nld.db.count_games('taster-dataset', conn=db_conn)} games.")

dataset = nld.TtyrecDataset(
    "taster-dataset",
    batch_size=32,
    seq_length=32,
    dbfilename=dbfilename,
)

minibatch = next(iter(dataset))
minibatch.keys()

from nle.nethack import tty_render

batch_idx = 0
time_idx = 0
chars = minibatch['tty_chars'][batch_idx, time_idx]
colors = minibatch['tty_colors'][batch_idx, time_idx]
cursor = minibatch['tty_cursor'][batch_idx, time_idx]

print(tty_render(chars, colors, cursor))

import torch
from nle.env.tasks import NetHackChallenge


env = NetHackChallenge(
    savedir=None,  # Do not save any recordings.
    character='@', # Randomly rotate through characters.
)

# Then use the environment actions to convert the keypresses.
embed_actions = torch.zeros((256, 1))
for i, a in enumerate(env.actions):
    embed_actions[a.value][0] = i

embed_actions = torch.nn.Embedding.from_pretrained(embed_actions)
keypresses = torch.Tensor(minibatch["keypresses"]).long()
actions = embed_actions(keypresses).squeeze(-1).long()

shuffle_small_dataset = nld.TtyrecDataset(
    "taster-dataset",
    batch_size=2,
    seq_length=6000,
    dbfilename=dbfilename,
    shuffle=True,
    loop_forever=False,
    gameids=[34,550,45],
)
for epoch in range(3):
    print(f"Epoch: {epoch}")
    for ind, mb in enumerate(shuffle_small_dataset):
        gameids = mb["gameids"][:, 0]
        print(f"  Batch {ind} first timestep gameids: {gameids}")
    print()

# Build the subselect sql query
subselect_sql = "SELECT gameid FROM games WHERE role=? AND race=?"
subselect_sql_args = ("Mon", "Hum")
batch_size = 10

# Build the dataset
monk_dataset = nld.TtyrecDataset(
    "taster-dataset",
    batch_size=batch_size,
    seq_length=2,
    dbfilename=dbfilename,
    subselect_sql=subselect_sql,
    subselect_sql_args=subselect_sql_args
)

# See from the error how there are fewer than 10k games despite the full dataset having 109k
print(f"Full Dataset has {nld.db.count_games('taster-dataset', conn=db_conn):,} games.")
print(f"Human Monk Subdataset Has: {len(monk_dataset._gameids)} games")

mb = next(iter(monk_dataset))

batch_idx = 0
time_idx = 0
chars = mb['tty_chars'][batch_idx, time_idx]
colors = mb['tty_colors'][batch_idx, time_idx]
cursor = mb['tty_cursor'][batch_idx, time_idx]

print(tty_render(chars, colors, cursor))

from concurrent.futures import ThreadPoolExecutor
import time


with ThreadPoolExecutor(max_workers=10) as tp:
    dataset = nld.TtyrecDataset(
        "taster-dataset",
        batch_size=100,
        seq_length=100,
        dbfilename=dbfilename,
        threadpool=tp
    )
    start = time.time()
    for i, mb in enumerate(dataset):
        if i == 10:
            break
    end = time.time()
    chars = mb['tty_chars'][batch_idx, time_idx]
    colors = mb['tty_colors'][batch_idx, time_idx]
    cursor = mb['tty_cursor'][batch_idx, time_idx]

    print(tty_render(chars, colors, cursor))
# NB this might be v slow on free Colab, try on laptop or server.
print(f"Loaded 100,000 frames in {end-start:.2f}s")

dataset = nld.TtyrecDataset('taster-dataset', dbfilename=dbfilename)
mb = next(iter(dataset))
gameid = mb["gameids"][0][0]

chars = mb['tty_chars'][0, 0]
colors = mb['tty_colors'][0, 0]
cursor = mb['tty_cursor'][0, 0]

print(tty_render(chars, colors, cursor))

print(dict(dataset.get_meta(gameid)))

import gym
import nle
import nle.dataset as nld
from datetime import datetime

def generate_rollouts(env):
    obs = env.reset()
    episodes = 0
    while episodes < 10:
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
            episodes += 1

# 1. Create some envs, with a savedir directory 'path/to/save/X'
envA = gym.make("NetHackChallenge-v0", savedir="path/to/save/A", save_ttyrec_every=2)
envB = gym.make("NetHackScore-v0", character="Mon-Hum-Neu-Mal", savedir="path/to/save/B", save_ttyrec_every=1)

# 2. Generate rollouts
generate_rollouts(envA)
generate_rollouts(envB)

# 3. Add to directory, with given unique dataset name
name = f"dataset_{datetime.now().time()}"
if not nld.db.exists():
    nld.db.create()
nld.add_nledata_directory("path/to/save", name)

# 4. Use and enjoy!
dataset = nld.TtyrecDataset(name)
print(f"Dataset has {len(dataset._gameids)} entries!")

