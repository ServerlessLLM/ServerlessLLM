import os
import subprocess
import sys

import serverless_llm_store


def main():
    print("TODO Run server...")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        os.path.join(serverless_llm_store.__path__[0])
        + ":"
        + env.get("LD_LIBRARY_PATH", "")
    )

    sys.exit(
        subprocess.call(
            [
                os.path.join(
                    serverless_llm_store.__path__[0], "sllm_store_server"
                ),
                *sys.argv[1:],
            ],
            env=env,
        )
    )


if __name__ == "__main__":
    main()
