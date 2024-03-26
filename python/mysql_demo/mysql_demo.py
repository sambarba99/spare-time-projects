"""
MySQL demo

Author: Sam Barba
Created 18/08/2019
"""

import mysql.connector


DATABASE_NAME = 'company_database'
con = curs = None


def get_pwd():
	with open('C:/Users/Sam/Desktop/projects/sql_password.txt', 'r') as file:
		pwd = file.read()
	return pwd


def drop_database():
	con = mysql.connector.connect(host='localhost', user='user1', passwd=get_pwd())
	curs = con.cursor()
	curs.execute(f'DROP DATABASE IF EXISTS {DATABASE_NAME};')


def create_database():
	global con, curs

	con = mysql.connector.connect(host='localhost', user='user1', passwd=get_pwd())
	curs = con.cursor()
	curs.execute(f'CREATE DATABASE {DATABASE_NAME};')
	con = mysql.connector.connect(host='localhost', user='user1', passwd=get_pwd(), database=DATABASE_NAME)
	curs = con.cursor()


def create_tables():
	create_table_employee = 'CREATE TABLE employee (' \
		'employee_id INT PRIMARY KEY, ' \
		'surname VARCHAR(50), ' \
		'forename VARCHAR(50), ' \
		'birthday DATE, ' \
		'sex VARCHAR(1), ' \
		'salary INT, ' \
		'super_id INT, ' \
		'branch_id INT);'

	create_table_branch = 'CREATE TABLE branch (' \
		'branch_id INT PRIMARY KEY, ' \
		'branch_name VARCHAR(50), ' \
		'manager_id INT, ' \
		'manager_start_date DATE, ' \
		'FOREIGN KEY(manager_id) REFERENCES employee(employee_id) ON DELETE SET NULL);'

	alter_table_employee1 = 'ALTER TABLE employee ' \
		'ADD FOREIGN KEY(branch_id) REFERENCES branch(branch_id) ON DELETE SET NULL;'

	alter_table_employee2 = 'ALTER TABLE employee ' \
		'ADD FOREIGN KEY(super_id) REFERENCES employee(employee_id) ON DELETE SET NULL;'

	create_table_branch_supplier = 'CREATE TABLE branch_supplier (' \
		'branch_id INT, ' \
		'supplier_name VARCHAR(50), ' \
		'supply_type VARCHAR(50), ' \
		'PRIMARY KEY(branch_id, supplier_name), ' \
		'FOREIGN KEY(branch_id) REFERENCES branch(branch_id) ON DELETE CASCADE);'

	create_table_client = 'CREATE TABLE client (' \
		'client_id INT PRIMARY KEY, ' \
		'client_name VARCHAR(50), ' \
		'branch_id INT, ' \
		'FOREIGN KEY(branch_id) REFERENCES branch(branch_id) ON DELETE SET NULL);'

	create_table_works_with = 'CREATE TABLE works_with (' \
		'employee_id INT, ' \
		'client_id INT, ' \
		'total_sales INT, ' \
		'PRIMARY KEY(employee_id, client_id), ' \
		'FOREIGN KEY(employee_id) REFERENCES employee(employee_id) ON DELETE CASCADE, ' \
		'FOREIGN KEY(client_id) REFERENCES client(client_id) ON DELETE CASCADE);'

	curs.execute(create_table_employee)
	curs.execute(create_table_branch)
	curs.execute(alter_table_employee1)
	curs.execute(alter_table_employee2)
	curs.execute(create_table_branch_supplier)
	curs.execute(create_table_client)
	curs.execute(create_table_works_with)


def populate_tables():
	# 1. Tables 'employee' and 'branch'
	# Corporate
	curs.execute("INSERT INTO employee VALUES(100, 'Wallace', 'David', '1967-11-17', 'M', 250000, NULL, NULL);")
	curs.execute("INSERT INTO branch VALUES(1, 'Corporate', 100, '1993-05-18');")
	curs.execute('UPDATE employee SET branch_id = 1 WHERE employee_id = 100;')
	curs.execute("INSERT INTO employee VALUES(101, 'Levinson', 'Jan', '1961-05-11', 'F', 110000, 100, 1);")
	con.commit()

	# Scranton
	curs.execute("INSERT INTO employee VALUES(102, 'Scott', 'Michael', '1964-03-15', 'M', 75000, 100, NULL);")
	curs.execute("INSERT INTO branch VALUES(2, 'Scranton', 102, '1996-01-15');")
	curs.execute('UPDATE employee SET branch_id = 2 WHERE employee_id = 102;')
	curs.execute("INSERT INTO employee VALUES(103, 'Martin', 'Angela', '1971-06-25', 'F', 63000, 102, 2);")
	curs.execute("INSERT INTO employee VALUES(104, 'Kapoor', 'Kelly', '1980-02-05', 'F', 55000, 102, 2);")
	curs.execute("INSERT INTO employee VALUES(105, 'Hudson', 'Stanley', '1958-02-19', 'M', 69000, 102, 2);")
	con.commit()

	# Stamford
	curs.execute("INSERT INTO employee VALUES(106, 'Porter', 'Josh', '1969-09-05', 'M', 78000, 100, NULL);")
	curs.execute("INSERT INTO branch VALUES(3, 'Stamford', 106, '1998-02-13');")
	curs.execute('UPDATE employee SET branch_id = 3 WHERE employee_id = 106;')
	curs.execute("INSERT INTO employee VALUES(107, 'Bernard', 'Andy', '1973-07-22', 'M', 65000, 106, 3);")
	curs.execute("INSERT INTO employee VALUES(108, 'Halpert', 'Jim', '1978-10-01', 'M', 71000, 106, 3);")
	con.commit()

	# 2. Table 'branch_supplier'
	curs.execute("INSERT INTO branch_supplier VALUES(2, 'Hammer Mill', 'Paper');")
	curs.execute("INSERT INTO branch_supplier VALUES(2, 'Uni-ball', 'Writing Utensils');")
	curs.execute("INSERT INTO branch_supplier VALUES(3, 'Patriot Paper', 'Paper');")
	curs.execute("INSERT INTO branch_supplier VALUES(2, 'J.T. Forms & Labels', 'Custom Forms');")
	curs.execute("INSERT INTO branch_supplier VALUES(3, 'Uni-ball', 'Writing Utensils');")
	curs.execute("INSERT INTO branch_supplier VALUES(3, 'Hammer Mill', 'Paper');")
	curs.execute("INSERT INTO branch_supplier VALUES(3, 'Stamford Lables', 'Custom Forms');")
	con.commit()

	# 3. Table 'client'
	curs.execute("INSERT INTO client VALUES(400, 'Dunmore Highschool', 2);")
	curs.execute("INSERT INTO client VALUES(401, 'Lackawana County', 2);")
	curs.execute("INSERT INTO client VALUES(402, 'FedEx', 3);")
	curs.execute("INSERT INTO client VALUES(403, 'John Daly Law, LLC', 3);")
	curs.execute("INSERT INTO client VALUES(404, 'Scranton Whitepages', 2);")
	curs.execute("INSERT INTO client VALUES(405, 'Times Newspaper', 3);")
	curs.execute("INSERT INTO client VALUES(406, 'FedEx', 2);")
	con.commit()

	# 4. Table 'works_with'
	curs.execute('INSERT INTO works_with VALUES(105, 400, 55000);')
	curs.execute('INSERT INTO works_with VALUES(102, 401, 267000);')
	curs.execute('INSERT INTO works_with VALUES(108, 402, 22500);')
	curs.execute('INSERT INTO works_with VALUES(107, 403, 5000);')
	curs.execute('INSERT INTO works_with VALUES(108, 403, 12000);')
	curs.execute('INSERT INTO works_with VALUES(105, 404, 33000);')
	curs.execute('INSERT INTO works_with VALUES(107, 405, 26000);')
	curs.execute('INSERT INTO works_with VALUES(102, 406, 15000);')
	curs.execute('INSERT INTO works_with VALUES(105, 406, 130000);')
	con.commit()


if __name__ == '__main__':
	drop_database()
	create_database()
	create_tables()
	populate_tables()

	# ----- Plain select statements -----

	statements = [
		'DESCRIBE employee;',
		'SELECT * FROM employee;',
		'SELECT forename, surname FROM employee LIMIT 4;',
		'SELECT * FROM employee WHERE salary >= 75000 ORDER BY sex DESC, birthday;',   # Employees with at least 75k salary
		'SELECT COUNT(employee_id) FROM employee;',                                    # Number of employees
		'SELECT DISTINCT sex FROM employee;',                                          # Unique sex values
		'SELECT sex, COUNT(sex) FROM employee GROUP BY sex;',                          # Counts of each unique sex
		'SELECT sex, ROUND(AVG(salary), 2) FROM employee GROUP BY sex;',               # Average salary of each sex
		'SELECT employee_id, SUM(total_sales) FROM works_with GROUP BY employee_id;',  # Total sales of each salesperson
		'SELECT client_id, SUM(total_sales) FROM works_with GROUP BY client_id;',      # Total money spent by each client
		"SELECT * FROM client WHERE client_name LIKE '%LLC%';",                        # Clients that are LLCs
		'SELECT forename FROM employee UNION SELECT branch_name FROM branch;'          # Employee and branch names
	]

	for statement in statements:
		print(f'>>> {statement}')
		print('Result:')
		curs.execute(statement)
		for i in curs:
			print(f'\t{i}')
		print()

	# ----- Nested select statements -----

	# Find names of all employees who have sold over 25k
	statement = 'SELECT employee.forename, employee.surname ' \
		'FROM employee ' \
		'WHERE employee.employee_id IN ' \
		'(SELECT works_with.employee_id FROM works_with WHERE works_with.total_sales > 25000);'
	print(f'>>> {statement}')
	print('Result:')
	curs.execute(statement)
	for i in curs:
		print(f'\t{i}')

	# ----- Join statements -----

	# Find all branches and names of their managers
	statement = 'SELECT employee.employee_id, employee.forename, employee.surname, branch.branch_name ' \
		'FROM employee ' \
		'JOIN branch ON employee.employee_id = branch.manager_id;'  # Inner join
	print(f'\n>>> {statement}')
	print('Result:')
	curs.execute(statement)
	for i in curs:
		print(f'\t{i}')
