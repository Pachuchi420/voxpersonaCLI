import argparse
import asyncio

def main():
    parser = argparse.ArgumentParser(prog="voxPersona")
    parser.add_argument("-socket", action="store_true",
                        help="Run the legacy scripted-dialogue websocket backend (core/BackService_ws.py), on port 8765")
    parser.add_argument("-host", default="127.0.0.1", help="Web app bind address (use 0.0.0.0 to expose beyond localhost)")
    parser.add_argument("-port", type=int, default=8000, help="Web app port")
    args = parser.parse_args()

    if args.socket:
        from core.BackService_ws import main as run_socket_server
        asyncio.run(run_socket_server())
    else:
        import uvicorn
        uvicorn.run("webapp.server:app", host=args.host, port=args.port)

if __name__ == "__main__":
    main()



