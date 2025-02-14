# app.py
from primary_agent import get_primary_agent
from smolagents import GradioUI

# Get the primary agent from the primary_agent.py
primary_agent = get_primary_agent()


# Main function to launch UI
def main():
    GradioUI(primary_agent).launch()


if __name__ == "__main__":
    main()
