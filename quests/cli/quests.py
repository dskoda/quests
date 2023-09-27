import click

from quests.cli.make_descriptors import make_descriptors
from quests.cli.entropy import entropy
from quests.cli.entropy_sampler import entropy_sampler


class QuestsGroup(click.Group):
    pass


@click.command(cls=QuestsGroup)
def quests():
    """Command line interface for quests"""


quests.add_command(entropy)
quests.add_command(entropy_sampler)
quests.add_command(make_descriptors)


if __name__ == "__main__":
    quests()
