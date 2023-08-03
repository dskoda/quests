import click

from quests.cli.compare import compare
from quests.cli.entropy import entropy
from quests.cli.dentropy import dentropy


class QuestsGroup(click.Group):
    pass


@click.command(cls=QuestsGroup)
def quests():
    """Command line interface for mkite_core"""


quests.add_command(compare)
quests.add_command(entropy)
quests.add_command(dentropy)


if __name__ == "__main__":
    quests()
