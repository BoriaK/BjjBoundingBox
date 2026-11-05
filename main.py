# Entry point to run the detector workflows
import argparse

from run_Detector_w_Screenshot_v4 import main as run_v4
from run_Detector_w_Screenshot_v5 import main as run_v5
from run_Detector_Dynamic_Screenshot import main as run_dynamic_screenshot


def main():
    parser = argparse.ArgumentParser(description="Run a BJJ detector workflow.")
    parser.add_argument(
        "--mode",
        choices=["screenshot", "combined", "dynamic"],
        default="combined",
        help="'screenshot' = static screenshot detector (v4, default). "
             "'combined' = v5: Insert for classical screenshot, Shift+Insert for dynamic recording. "
             "'dynamic' = standalone short screen-recording dynamic screenshot detector.",
    )
    args = parser.parse_args()

    if args.mode == "combined":
        run_v5()
    elif args.mode == "dynamic":
        run_dynamic_screenshot()
    else:
        run_v4()


if __name__ == '__main__':
    main()
