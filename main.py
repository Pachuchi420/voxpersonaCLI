import argparse
import asyncio

def main(): 
    parser = argparse.ArgumentParser(prog="voxPersona")
    parser.add_argument("-socket", action="store_true",
                        help="Run voxPersona on a websocket, on port 8765")
    args = parser.parse_args()

    if args.socket: 
        from core.BackService_ws import main as run_socket_server
        asyncio.run(run_socket_server())
    else:
        from tui.app import ConvoTUI
        ConvoTUI().run()

if __name__ == "__main__":
    main()



