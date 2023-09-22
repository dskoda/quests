import click

from quests.cli.compare import compare
from quests.cli.entropy_flex import entropy_flex
from quests.cli.entropy_sampler import entropy_sampler
from quests.cli.dentropy import dentropy


class QuestsGroup(click.Group):
    pass


@click.command(cls=QuestsGroup)
def quests():
    """Command line interface for mkite_core"""


quests.add_command(compare)
quests.add_command(entropy_flex)
quests.add_command(entropy_sampler)
quests.add_command(dentropy)


if __name__ == "__main__":
    quests()
