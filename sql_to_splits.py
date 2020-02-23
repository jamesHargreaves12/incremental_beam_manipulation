import sqlite3

# connecting to the database
connection = sqlite3.connect("data/datasets/nl_to_sql/text2sql-data/data/advising-db.sql")

# cursor
crsr = connection.cursor()

sql_command = """INSERT INTO emp VALUES (23, "Rishabh", "Bansal", "M", "2014-03-28");"""
crsr.execute(sql_command)


# SQL command to create a table in the database
# sql_command = """CREATE TABLE emp (
# staff_number INTEGER PRIMARY KEY,
# fname VARCHAR(20),
# lname VARCHAR(30),
# gender CHAR(1),
# joining DATE);"""
