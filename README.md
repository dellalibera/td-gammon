
<h1 align="center">TD-Gammon</h1> <br>
<p align="center">
   <img alt="Backgammon" title="Backgammon" src="logo.png" width="350">
</p>

---
# Table of Contents
- [Features](#features)
- [Installation](#installation)
- [How to interact with GNU Backgammon using Python Script?](#howto)
- [Usage](#usage)
    - [Train TD-Network](#train)
    - [Evaluate Agent(s)](#evaluate)
    - [Web Interface](#web_interface)
    - [Plot Wins](#plot)
- [Backgammon OpenAI Gym Environment](#env)
- [Bibliography, sources of inspiration, related works](#biblio)
- [License](#license)
---
## <a name="features"></a>Features
- PyTorch implementation of TD-Gammon [1].
- Test the trained agents against an open source implementation of the Backgammon game, [GNU Backgammon](https://www.gnu.org/software/gnubg/).
- Play against a trained agent via web gui

---
## <a name="installation"></a>Installation

I used [`Anaconda3`](https://www.anaconda.com/distribution/), with `Python 3.6.8` (I tested only with the following configurations). 

Create the conda environment: 
```
$ conda create --name tdgammon python=3.6
$ source activate tdgammon
(tdgammon) $ git clone https://github.com/dellalibera/td-gammon.git
```
Install the environment [`gym-backgammon`](#https://github.com/dellalibera/gym-backgammon):
```
(tdgammon) $ git clone https://github.com/dellalibera/gym-backgammon.git
(tdgammon) $ cd gym-backgammon
(tdgammon) $ pip install -e .
```

Install the dependencies [`pytorch v1.2`](https://pytorch.org/get-started/locally/):
```
(tdgammon) $ pip install torch torchvision
(tdgammon) $ pip install tb-nightly
```
or
```
(tdgammon) $ cd td-gammon/
(tdgammon) $ pip install -r requirements.txt
```

####  Without Anaconda Environment
If you don't use Anaconda environment, run the following commands:
```
git clone https://github.com/dellalibera/td-gammon.git
pip3 install -r td-gammon/requirements.txt
git clone https://github.com/dellalibera/gym-backgammon.git
cd gym-backgammon/
pip3 install -e .
```
If you don't use Anaconda environment, in the commands below replace `python` with `python3`.


### GNU Backgammon
To play against `gnubg`, you have to install [`gnubg`](https://www.gnu.org/software/gnubg/).  
**NOTE**: I installed `gnubg` on `Ubuntu 18.04` (running on a Virtual Machine), with `Python 2.7` (see next section to see how to interact with GNU Backgammon).  
#### On Ubuntu:
```
sudo apt-get install gnubg
```
---
## <a name="howto"></a>How to interact with GNU Backgammon using Python Script?
I used an `http server` that runs on the Guest machine (Ubuntu), to receive commands and interact with the `gnubg` program.  
In this way, it's possible to send commands from the Host machine (in my case `MacOS`).  
<br>
The file `bridge.py` should be executed on the Guest Machine (the machine where `gnubg` is installed).
#### On Ubuntu:
```
gnubg -t -p /path/to/bridge.py
```
It runs the `gnubg` with the command-line instead of using the graphical interface (`-t`) and evaluates a Python code file and exits (`-p`).  
For a list of parameters of `gnubg`, run `gnubg --help`.   
<br>
The python script `bridge.py` creates an `http server`, running on `localhost:8001`.  
If you want to modify the host and the port, change the following line in `bridge.py`:
```python
if __name__ == "__main__":
    HOST = 'localhost' # <-- YOUR HOST HERE
    PORT = 8001  # <-- YOUR PORT HERE
    run(host=HOST, port=PORT)
```
The file `td_gammon/gnubg/gnubg_backgammon.py` sends messages/commands to `gnubg` and parses the response.

---
## <a name="usage"></a>Usage
Run `python /path/to/main.py --help` for a list of parameters.

### <a name="train"></a>Train TD-Network
To train a neural network with a single layer with `40` hidden units, for `100000` games/episodes and save the model every `10000`, run the following command:
```
(tdgammon) $ python /path/to/main.py train --save_path ./saved_models/exp1 --save_step 10000 --episodes 100000 --name exp1 --type nn --lr 0.1 --hidden_units 40
```
Run `python /path/to/main.py train --help` for a list of parameters available for training.

--- 
### <a name="evaluate"></a>Evaluate Agent(s)
To evaluate an already trained models, you have to options: evaluate models to play against each other or evaluate one model against `gnubg`.  
Run `python /path/to/main.py evaluate --help` for a list of parameters available for evaluation.


### Agent vs Agent
To evaluate two model to play against each other you have to specify the path where the models are saved with the corresponding number of hidden units.
```
(tdgammon) $ python /path/to/main.py evaluate --episodes 50 --hidden_units_agent0 40 --hidden_units_agent1 40 --type nn --model_agent0 path/to/saved_models/agent0.tar --model_agent1 path/to/saved_models/agent1.tar
```

### Agent vs gnubg
To evaluate one model to play against `gnubg`, first you have to run `gnubg` with the script `bridge` as input.   
On Ubuntu (or where `gnubg` is installed)
```
gnubg -t -p /path/to/bridge.py
```
Then run (to play vs `gnubg` at intermediate level for 100 games):
```
(tdgammon) $ python /path/to/main.py evaluate --episodes 50 --hidden_units_agent0 40 --type nn --model_agent0 path/to/saved_models/agent0.tar vs_gnubg --difficulty beginner --host GNUBG_HOST --port GNUBG_PORT
```
The hidden units (`--hidden_units_agent0`) of the model must be same of the loaded model (`--model_agent0`).

--- 
### <a name="web_interface"></a>Web Interface
You can play against a trained agent via a web gui:
```
(tdgammon) $ python /path/to/main.py gui --host localhost --port 8002 --model path/to/saved_models/agent0.tar --hidden_units 40 --type nn
```
Then navigate to `http://localhost:8002` in your browser:
<p align="center">
   <img alt="Web Interface" title="Web Interface" src="gui_example.png">
</p>

Run `python /path/to/main.py gui --help` for a list of parameters available about the web gui.

--- 
### <a name="plot"></a>Plot Wins
Instead of evaluating the agent during training (it can require some time especially if you evaluate against `gnubg` - difficulty `world_class`), you can load all the saved models in a folder, and evaluate each model (saved at different time during training) against one or more opponents.  
The models in the directory should be of the same type (i.e the structure of the network should be the same for all the models in the same folder).

To plot the wins against `gnubg`, run on Ubuntu (or where `gnubg` is installed):
```
gnubg -t -p /path/to/bridge.py
```
In the example below the trained model is going to be evaluated against `gnubg` on two different difficulties levels - `beginner` and `advanced`:`
```
(tdgammon) $ python /path/to/main.py plot --save_path /path/to/saved_models/myexp --hidden_units 40 --episodes 10 --opponent random,gnubg --dst /path/to/experiments --type nn --difficulty beginner,advanced --host GNUBG_HOST --port GNUBG_PORT
```
To visualize the plots:
```
(tdgammon) $ tensorboard --logdir=runs/path/to/experiment/ --host localhost --port 8001
```
Run `python /path/to/main.py plot --help` for a list of parameters available about plotting.

## <a name="env"></a>Backgammon OpenAI Gym Environment
For a detailed description of the environment: [`gym-backgammon`](https://github.com/dellalibera/gym-backgammon).

---
## <a name="biblio"></a>Bibliography, sources of inspiration, related works
- TD-Gammon and Temporal Difference Learning:
    - [1] [Practical Issues in Temporal Difference Learning](https://papers.nips.cc/paper/465-practical-issues-in-temporal-difference-learning.pdf)
    - [Temporal Difference Learning and TD-Gammon](https://researcher.watson.ibm.com/researcher/view_page.php?id=7021)
    - [Programming backgammon using self-teaching neural nets](www.bkgm.com/articles/tesauro/ProgrammingBackgammon.pdf)
    - [Implementaion Details TD-Gammon](http://www.scholarpedia.org/article/User:Gerald_Tesauro/Proposed/Td-gammon)
    - [Chapter 9 Temporal-Difference Learning](https://web.stanford.edu/group/pdplab/pdphandbook/handbookch10.html)
    - [Implementation Details of the TD(λ) Procedure for the Case of Vector Predictions and Backpropagation](https://www.ece.uvic.ca/~bctill/papers/learning/Sutton_1987.pdf)
    - [Learning to Predict by the Methods of Temporal Differences](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) 
<br><br>
- GNU Backgammon: https://www.gnu.org/software/gnubg/ 
<br><br>
- Rules of Backgammon:
    - www.bkgm.com/rules.html
    - https://en.wikipedia.org/wiki/Backgammon
    - <a name="starting_position"></a>Starting Position: http://www.bkgm.com/gloss/lookup.cgi?starting+position
    - https://bkgm.com/faq/
<br><br>    
- Install GNU Backgammon on Ubuntu:
    - https://ubuntuforums.org/showthread.php?t=2217668
    - https://ubuntuforums.org/showthread.php?t=1506341
    - https://www.reddit.com/r/backgammon/comments/5gpkov/installing_gnu_or_xg_on_linux/
<br><br>
- How to use python to interact with `gnubg`: [\[Bug-gnubg\] Documentation: Looking for documentation on python scripting](https://www.mail-archive.com/bug-gnubg@gnu.org/msg06794.html)
<br><br>
- Other Implementation of the Backgammon OpenAI Gym Environment: 
    - https://github.com/edusta/gym-backgammon
<br><br>
- Other Implementation of TD-Gammon:
    - https://github.com/TobiasVogt/TD-Gammon
    - https://github.com/millerm/TD-Gammon
    - https://github.com/fomorians/td-gammon
<br><br>
- How to setup your VMWare Fusion images to use static IP addresses on Mac OS X
    - https://gist.github.com/pjkelly/1068716/6d19faa0122c0e1efe350e818bb8f4e8687ea1ab
<br><br>
- PyTorch Tensorboard: https://pytorch.org/docs/stable/tensorboard.html

---
## <a name="license"></a>License
[MIT](https://github.com/dellalibera/td-gammon/blob/master/LICENSE) 