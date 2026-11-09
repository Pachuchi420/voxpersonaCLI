import argparse
import asyncio
import os
import sys


def _fix_broken_libexpat_link() -> None:
    """Works around a broken dynamic-link on this kind of machine: some
    Homebrew python@3.12 builds link pyexpat.so against
    /usr/lib/libexpat.1.dylib, but if the system's copy is missing a
    symbol the build expects (a Homebrew-bottle/macOS version mismatch),
    importing plistlib raises ImportError -- which silently breaks
    anything that needs it transitively (pip's own wheel-tag resolution,
    funasr's deps for the emotion2vec backend, etc). Homebrew's own
    `expat` package has the missing symbol; pointing the dynamic linker
    at it fixes this, but DYLD_LIBRARY_PATH only takes effect at process
    *launch* -- setting it from inside an already-running process has no
    effect -- so if this process is affected, re-exec once with it set
    rather than silently limping along broken.
    """
    if sys.platform != "darwin" or os.environ.get("_VOXPERSONA_EXPAT_FIX_APPLIED"):
        return

    import platform
    if platform.mac_ver()[0]:
        return  # not affected on this machine

    expat_lib = "/opt/homebrew/opt/expat/lib"
    if not os.path.isdir(expat_lib):
        return  # nothing we can do -- not a Homebrew machine, or expat isn't installed

    env = os.environ.copy()
    existing = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{expat_lib}:{existing}" if existing else expat_lib
    env["_VOXPERSONA_EXPAT_FIX_APPLIED"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_fix_broken_libexpat_link()


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



