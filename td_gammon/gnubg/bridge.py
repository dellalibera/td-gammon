# THIS FILE SHOULD BE RUN ON THE SAME MACHINE WHERE gnubg IS INSTALLED.
# IT USES PYTHON 2.7

import gnubg
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
# from http.server import BaseHTTPRequestHandler, HTTPServer
import json

try:
    from urllib.parse import urlparse, parse_qs
except ImportError:
     from urlparse import urlparse, parse_qs


class Handler(BaseHTTPRequestHandler):

    def _set_headers(self, response=200):
        self.send_response(response)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        response = {'board': [], 'last_move': [], 'info': []}
        post_data = self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8')
        data = parse_qs(post_data)

        command = data['command'][0]
        print(command)

        prev_game = gnubg.match(0)['games'][-1]['game'] if gnubg.match(0) else []

        gnubg.command(command)

        # check if the game is started/exists (handle the case the command executed is set at the beginning)
        if gnubg.match(0):
            # get the board after the execution of a move
            response['board'] = gnubg.board()

            # get the last games
            games = gnubg.match(0)['games'][-1]

            # get the last game
            game = games['game'][-1]

            # save the state of the game before and after having executed a command
            response['last_move'] = [prev_game, game]

            # save the info al all games played so far
            for idx, g in enumerate(gnubg.match(0)['games']):
                info = g['info']

                response['info'].append(
                    {
                        'winner': info['winner'],
                        'n_moves': len(g['game']),
                        'resigned': info['resigned'] if 'resigned' in info else None
                    }
                )

        self._set_headers()
        self.wfile.write(json.dumps(response))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if self.path:
            self._set_headers()
            self.wfile.write(bytes("Hello! Welcome to Backgammon WebGUI"))


def run(host, server_class=HTTPServer, handler_class=Handler, port=8001):
    server_address = (host, port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd ({}:{})...'.format(host, port))
    httpd.serve_forever()


if __name__ == "__main__":
    HOST = 'localhost'  # <-- YOUR HOST HERE
    PORT = 8001  # <-- YOUR PORT HERE
    run(host=HOST, port=PORT)
