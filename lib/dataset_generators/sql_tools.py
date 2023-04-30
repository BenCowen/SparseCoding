import mysql.connector
from mysql.connector import Error
import pandas as pd
from non_committed_functions import get_sql_password


class Table:
    def __init__(self, name, column_names, column_details):
        self.name = name
        self.col_names = column_names
        self.col_deets = column_details

class DataBase:
    def __init__(self, database_name,
                 host_name='localhost',  # 3306,
                 user_name='root',
                 user_password='use_builtin'):

        self.db = None

        # Get password from uncommitted function containing sensitive info
        if user_password == 'use_builtin':
            user_password = get_sql_password()

        try:
            self.db = mysql.connector.connect(
                host=host_name,
                user=user_name,
                passwd=user_password,
                database=database_name
            )
            print(f"MySQL connection to database {database_name} successful")
            self.cursor = self.db.cursor()
        except Error as err:
            print(f"Failed to connect with database {database_name}." +
                  f"\n Error message: '{err}'")

    def add_table(self, table):
        command = f'CREATE TABLE IF NOT EXISTS {table.name} ('
        for col_name, col_deet in zip(table.col_names, table.col_deets):
            command += f' {col_name} {col_deet},'
        # Throw away last comma...
        command = command[:-1] + ');'
        self.cursor.execute(command)

    def extend_table(self, table, data_dict):
        """
        Takes a table and a dict of data where each key should correspond to a
            column of the table, and contain a list of data points to add.
        """
        # Initialize the insertion command:
        command = f"INSERT INTO {table.name} ("
        for name in table.col_names:
            command += f" {name},"
        command = command[:-1] + ")  VALUES "

        # Convert data to string:

        row_str = '('

        if type(data_dict[name]) == list:
            n_rows = len(data_dict[name])
            for row in range(n_rows):
                for key in data_dict.keys():
                    row_str += str(data_dict[key][row])
                    command += row_str + ')'
                    row_str = ', ('
        else:
            for key, val in data_dict.items():
                if type(val) == str:
                    val = f'\'{data_dict[key]}\''
                else:
                    val = str(val)
                row_str += str(val) + ','
            command += row_str[:-1] + ')'
        # Finally, insert into the table:
        self.cursor.execute(command)
        self.db.commit()

    def execute_query(self, query):
        try:
            self.cursor(query)
            self.db.commit()
        except Error as err:
            print(f'Query failed with error message {err}...')


if __name__ == "__main__":
    database = DataBase(database_name='inaccessible_worlds',
                        user_password=get_sql_password())
