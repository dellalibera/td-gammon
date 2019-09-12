import os
import sys
import json

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from gym_backgammon.envs.backgammon import WHITE, BLACK

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

COLORS = {WHITE: "White", BLACK: 'Black'}


class Handler(BaseHTTPRequestHandler):

    def parse_data(self, data):
        command = data['command'].lower()

        if command in ['start', 'new game']:
            response = self.server.dispatcher['start']()

        elif command in ['roll']:
            response = self.server.dispatcher['roll']()

        elif 'move' in command:
            response = self.server.dispatcher['move'](command)

        else:
            message = 'Invalid command\n'
            response = {'message': message, 'state': self.server.env.game.state, 'actions': self.server.last_commands}

        self.server.last_commands = response['actions']
        response['message'] = response['message'][:-1]  # remove the new line character of the last line

        return response

    def _set_headers(self, response=200):
        self.send_response(response)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        post_data = self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8')
        data = json.loads(post_data)
        # print(data)
        response = self.parse_data(data)
        self._set_headers()
        self.wfile.write(bytes(json.dumps(response), encoding='utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if self.path == '/':
            self._set_headers()
            # f = open("../td_gammon/web_gui/index.html").read()
            f = open(os.path.dirname(__file__) + "/../web_gui/index.html").read()

            # RESET THE STATE EVERY TIME A NEW PAGE IS REFRESHED
            self.server.reset()

            self.wfile.write(bytes(f, encoding='utf-8'))


class GUI:
    def __init__(self, env, host="localhost", port=8002, agents=None):
        self.host = host
        self.port = port
        self.server = HTTPServer((host, port), Handler)

        # GAME VARIABLES
        self.server.agents = agents
        self.server.env = env

        self.server.reset = self.reset
        self.server.reset()

        self.server.dispatcher = {
            'start': self.handle_start,
            'roll': self.handle_roll,
            'move': self.handle_move
        }

    def reset(self):
        self.server.agent = None
        self.server.first_roll = None
        self.server.wins = {WHITE: 0, BLACK: 0}
        self.server.roll = None
        self.server.game_started = False
        self.server.game_finished = False
        self.server.last_commands = []

    def handle_start(self):
        self.server.reset()
        # response = {'message': '', 'state': [], 'actions': []}
        message = '\nNew game started\n'
        commands = []

        if self.server.game_started:
            message = "The game is already started. To start a new game, type 'new game'\n"
            commands.append('new game')

        else:
            self.server.game_finished = False
            self.server.game_started = True

            agent_color, self.server.first_roll, observation = self.server.env.reset()
            self.server.agent = self.server.agents[agent_color]

            if agent_color == WHITE:
                message += "{} Starts first | Roll={} | Run 'move (src/target)'\n".format(COLORS[self.server.agent.color], (abs(self.server.first_roll[0]), abs(self.server.first_roll[1])))
                commands.extend(self.server.env.get_valid_actions(self.server.first_roll))
                self.server.roll = self.server.first_roll

            else:
                opponent = self.server.agents[agent_color]
                message += "{} Starts first | Roll={}\n".format(COLORS[opponent.color], (abs(self.server.first_roll[0]), abs(self.server.first_roll[1])))

                if self.server.first_roll:
                    roll = self.server.first_roll
                    self.server.first_roll = None
                else:
                    roll = opponent.roll_dice()

                actions = self.server.env.get_valid_actions(roll)
                action = opponent.choose_best_action(actions, self.server.env)
                message += "{} | Roll={} | Action={} | Run 'roll'\n".format(COLORS[opponent.color], roll, action)
                commands.extend(['roll', 'new game'])
                observation_next, reward, done, info = self.server.env.step(action)

                agent_color = self.server.env.get_opponent_agent()
                self.server.agent = self.server.agents[agent_color]

        return {'message': message, 'state': self.server.env.game.state, 'actions': list(commands)}

    def handle_roll(self):
        message = ''
        commands = []

        if self.server.roll is not None:
            message += "You have already rolled the dice {}. Run 'move (src/target)'\n".format((abs(self.server.roll[0]), abs(self.server.roll[1])))
            actions = self.server.env.get_valid_actions(self.server.roll)
            if len(actions) == 0:
                commands.append('start')
            else:
                commands.extend(list(actions))

        elif self.server.game_finished:
            message += "The game is finished. Type 'Start' to start a new game\n".format((abs(self.server.roll[0]), abs(self.server.roll[1])))
            commands.append('start')

        elif not self.server.game_started:
            message += "The game is not started. Type 'start' to start a new game\n"
            commands.append('start')

        else:
            self.server.roll = self.server.agent.roll_dice()
            message += "{} | Roll={} | Run 'move (src/target)'\n".format(COLORS[self.server.agent.color], (abs(self.server.roll[0]), abs(self.server.roll[1])))
            actions = self.server.env.get_valid_actions(self.server.roll)
            commands.extend(list(actions))

            if len(actions) == 0:
                message += "You cannot move\n"

                agent_color = self.server.env.get_opponent_agent()
                opponent = self.server.agents[agent_color]

                roll = opponent.roll_dice()

                actions = self.server.env.get_valid_actions(roll)
                action = opponent.choose_best_action(actions, self.server.env)
                message += "{} | Roll={} | Action={}\n".format(COLORS[opponent.color], roll, action)
                observation_next, reward, done, info = self.server.env.step(action)

                if done:
                    winner = self.server.env.game.get_winner()
                    message += "Game Finished!!! {} wins \n".format(COLORS[winner])
                    commands.append('new game')
                    self.server.game_finished = True
                else:
                    agent_color = self.server.env.get_opponent_agent()
                    self.server.agent = self.server.agents[agent_color]
                    self.server.roll = None
                    commands.extend(['roll', 'new game'])

        return {'message': message, 'state': self.server.env.game.state, 'actions': list(commands)}

    def handle_move(self, command):
        message = ''
        commands = []

        if self.server.roll is None:
            message += "You must roll the dice first\n"
            commands = self.server.last_commands

        elif self.server.game_finished:
            message += "The game is finished. Type 'new game' to start a new game\n".format((abs(self.server.roll[0]), abs(self.server.roll[1])))
            commands.append('new game')

        else:
            try:
                action = command.split()[1]
                action = action.split(',')
                play = []
                is_bar = False

                for move in action:
                    move = move.replace('(', '')
                    move = move.replace(')', '')
                    s, t = move.split('/')

                    if s == 'BAR' or s == 'bar':
                        play.append(('bar', int(t)))
                        is_bar = True
                    else:
                        play.append((int(s), int(t)))

                if is_bar:
                    action = tuple(play)
                else:
                    action = tuple(sorted(play, reverse=True))

            except Exception as e:
                message += "Error during parsing move\n"
                commands = self.server.last_commands

            else:
                actions = self.server.env.get_valid_actions(self.server.roll)

                if action not in actions:
                    message += "Illegal move | Roll={}\n".format((abs(self.server.roll[0]), abs(self.server.roll[1])))
                else:
                    message += "{} | Roll={} | Action={}\n".format(COLORS[self.server.agent.color], (abs(self.server.roll[0]), abs(self.server.roll[1])), action)
                    observation_next, reward, done, info = self.server.env.step(action)

                    if done:
                        winner = self.server.env.game.get_winner()
                        message += "Game Finished!!! {} wins\n".format(COLORS[winner])
                        commands.append('new game')
                        self.server.game_finished = True

                    else:
                        agent_color = self.server.env.get_opponent_agent()
                        opponent = self.server.agents[agent_color]

                        roll = opponent.roll_dice()
                        actions = self.server.env.get_valid_actions(roll)
                        action = opponent.choose_best_action(actions, self.server.env)

                        message += "{} | Roll={} | Action={}\n".format(COLORS[opponent.color], roll, action)
                        observation_next, reward, done, info = self.server.env.step(action)

                        if done:
                            winner = self.server.env.game.get_winner()
                            message += "Game Finished!!! {} wins\n".format(COLORS[winner])
                            commands.append('new game')
                            self.server.game_finished = True

                        else:
                            commands.extend(['roll', 'new game'])
                            agent_color = self.server.env.get_opponent_agent()
                            self.server.agent = self.server.agents[agent_color]
                            self.server.roll = None

        return {'message': message, 'state': self.server.env.game.state, 'actions': list(commands)}

    def run(self):
        print('Starting httpd (http://{}:{})...'.format(self.host, self.port))
        self.server.serve_forever()
