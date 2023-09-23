import click

from quests.cli.entropy import entropy
from quests.cli.entropy_sampler import entropy_sampler


class QuestsGroup(click.Group):
    pass


@click.command(cls=QuestsGroup)
def quests():
    """Command line interface for quests"""


quests.add_command(entropy)
quests.add_command(entropy_sampler)


if __name__ == "__main__":
    quests()
