import argparse
import utils


def formatter(prog):
    return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=100, width=180)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD-Gammon', formatter_class=lambda prog: formatter(prog))
    subparsers = parser.add_subparsers(help='Train TD-Network | Evaluate Agent(s) | Web GUI | Plot Wins')

    parser_train = subparsers.add_parser('train', help='Train TD-Network', formatter_class=lambda prog: formatter(prog))
    parser_train.add_argument('--save_path', help='Save directory location', type=str, default=None)
    parser_train.add_argument('--save_step', help='Save the model every n episodes/games', type=int, default=0)
    parser_train.add_argument('--episodes', help='Number of episodes/games', type=int, default=200000)
    parser_train.add_argument('--init_weights', help='Init Weights with zeros', action='store_true')
    parser_train.add_argument('--lr', help='Learning rate', type=float, default=1e-4)
    parser_train.add_argument('--hidden_units', help='Hidden units', type=int, default=40)
    parser_train.add_argument('--lamda', help='Credit assignment parameter', type=float, default=0.7)
    parser_train.add_argument('--model', help='Directory location to the model to be restored', type=str, default=None)
    parser_train.add_argument('--name', help='Name of the experiment', type=str, default='exp1')
    parser_train.add_argument('--type', help='Model type', choices=['cnn', 'nn'], type=str, default='nn')
    parser_train.add_argument('--seed', help='Seed used to reproduce results', type=int, default=123)

    parser_train.set_defaults(func=utils.args_train)

    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate Agent(s)', formatter_class=lambda prog: formatter(prog))
    parser_evaluate.add_argument('--model_agent0', help='Saved model used by the agent0 (WHITE)', required=True, type=str)
    parser_evaluate.add_argument('--model_agent1', help='Saved model used by the agent1 (BLACK)', required=False, type=str)
    parser_evaluate.add_argument('--type', help='Model type used by the agents', choices=['cnn', 'nn'], type=str, default='nn')
    parser_evaluate.add_argument('--hidden_units_agent0', help='Hidden Units of the model used by the agent0 (WHITE)', required=False, type=int, default=40)
    parser_evaluate.add_argument('--hidden_units_agent1', help='Hidden Units of the model used by the agent1 (BLACK)', required=False, type=int, default=40)
    parser_evaluate.add_argument('--episodes', help='Number of episodes/games', default=20, required=False, type=int)

    subparsers_gnubg = parser_evaluate.add_subparsers(help='Parameters for gnubg interface')
    parser_gnubg = subparsers_gnubg.add_parser('vs_gnubg', help='Evaluate agent0 against gnubg', formatter_class=lambda prog: formatter(prog))
    parser_gnubg.add_argument('--host', help='Host running gnubg', type=str, required=True)
    parser_gnubg.add_argument('--port', help='Port listening for gnubg commands', type=int, required=True)
    parser_gnubg.add_argument('--difficulty', help='Difficulty level', choices=['beginner', 'intermediate', 'advanced', 'world_class'], type=str, required=False, default='beginner')

    parser_gnubg.set_defaults(func=utils.args_gnubg)
    parser_evaluate.set_defaults(func=utils.args_evaluate)

    parser_gui = subparsers.add_parser('gui', help='Start Web GUI', formatter_class=lambda prog: formatter(prog))
    parser_gui.add_argument('--host', help='Host running the web gui', default='localhost')
    parser_gui.add_argument('--port', help='Port listening for command', default=8002, type=int)
    parser_gui.add_argument('--model', help='Model used by the AI opponent', required=True, type=str)
    parser_gui.add_argument('--hidden_units', help='Hidden units of the model loaded', required=False, type=int, default=40)
    parser_gui.add_argument('--type', help='Model type', choices=['cnn', 'nn'], type=str, default='nn')

    parser_gui.set_defaults(func=utils.args_gui)

    parser_plot = subparsers.add_parser('plot', help='Plot the performance (wins)', formatter_class=lambda prog: formatter(prog))
    parser_plot.add_argument('--save_path', help='Directory where the model are saved', type=str, required=True)
    parser_plot.add_argument('--hidden_units', help='Hidden units of the model(s) loaded', type=int, default=40)
    parser_plot.add_argument('--episodes', help='Number of episodes/games against a single opponent', default=20, type=int)
    parser_plot.add_argument('--opponent', help='Opponent(s) agent(s) (delimited by comma) - "random" and/or "gnubg"', default='random', type=str)
    parser_plot.add_argument('--host', help='Host running gnubg (if gnubg in --opponent)', type=str)
    parser_plot.add_argument('--port', help='Port listening for gnubg commands (if gnubg in --opponent)', type=int)
    parser_plot.add_argument('--difficulty', help='Difficulty level(s) (delimited by comma)', type=str, default="beginner,intermediate,advanced,world_class")
    parser_plot.add_argument('--dst', help='Save directory location', type=str, default='myexp')
    parser_plot.add_argument('--type', help='Model type', choices=['cnn', 'nn'], type=str, default='nn')

    parser_plot.set_defaults(func=lambda args: utils.args_plot(args, parser))

    args = parser.parse_args()
    args.func(args)
