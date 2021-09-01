# Calendar app
# Author: Sam Barba
# Created 11/11/2018

import datetime

CALENDAR_PATH = "C:\\Users\\Sam Barba\\Desktop\\Programs\\programoutputs\\calendar.csv"
DAYS_OF_WEEK = {0: "Monday",
	1: "Tuesday",
	2: "Wednesday",
	3: "Thursday",
	4: "Friday",
	5: "Saturday",
	6: "Sunday"}
MONTHS = {0: "January",
	1: "February",
	2: "March",
	3: "April",
	4: "May",
	5: "June",
	6: "July",
	7: "August",
	8: "September",
	9: "October",
	10: "November",
	11: "December"}

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Calendar:
	# No need for adding and removing methods, as inbuilt can be used
	# i.e. self.events.append(event)
	#      self.events.remove(event)
	#      self.events.clear()
	
	def __init__(self, events):
		self.events = events

	def __repr__(self):
		if not self.events: return "No events"

		cal = []

		for i in range(len(self.events)):
			event = str(i + 1) + ". " + str(self.events[i])
			cal.append(event)

		return "\n".join(cal)

	def sortEvents(self): # Sort events chronologically
		self.events.sort(key = lambda ev: ev.timeOld)

	def removePastEvents(self):
		currentDateTimeOld = daysSince112000(currentDate())

		self.events = list(filter(lambda ev: ev.timeOld > currentDateTimeOld, self.events))

		writeFile(self)

class Event:
	def __init__(self, date, title, timeOld):
		self.date = date
		self.title = title
		self.timeOld = timeOld

	def __repr__(self):
		d, m, y = map(int, self.date.split("/"))

		dayDate = DAYS_OF_WEEK[(self.timeOld - 2) % 7] + " " + str(d) + ordinal(d) + " " + MONTHS[m - 1] + " " + str(y)
		daysUntil = daysSince112000(self.date) - daysSince112000(currentDate())

		if daysUntil > 0:
			return "{}: {} (in {} days)".format(dayDate, self.title, daysUntil)
		elif daysUntil == 0:
			return "{}: {} (today)".format(dayDate, self.title)
		else:
			return "{}: {} ({} days ago)".format(dayDate, self.title, -daysUntil)

# ---------------------------------------------------------------------------------------------------- #
# -------------------------------------------  FUNCTIONS  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def intValid(userIn):
	try:
		n = int(userIn)
		return True
	except:
		return False

def dateValid(date):
	if len(date) != len("DD/MM/YYYY"): return False

	for i in range(len(date)):
		a = ord(date[i]) # ASCII value of current char

		if i in [2, 5] and a != 47:
			return False # 3rd and 6th chars should be '/'
		elif i not in [2, 5] and (a < 48 or a > 57):
			return False # Rest should be numeric

		d, m, y = map(int, date.split("/"))

		return (y >= 2000 and 1 <= m <= 12 and 1 <= d <= daysInMonth(m, y)
			and daysSince112000(date) > daysSince112000(currentDate()))

def daysInMonth(m, y):
	if m in [4, 6, 9, 11]:
		return 30
	elif m == 2:
		if ((y % 4 == 0) and (y % 100 != 0)) or y % 400 == 0: # If y is leap year
			return 29
		else:
			return 28
	else:
		return 31

# Days since 01/01/2000
def daysSince112000(date):
	d, m, y = map(int, date.split("/"))

	count, d1, m1, y1 = 0, 1, 1, 2000

	while d1 != d or m1 != m or y1 != y:
		count += 1
		d1 += 1

		if d1 > daysInMonth(m1, y1):
			d1 = 1
			m1 += 1
			if m1 > 12:
				m1 = 1
				y1 += 1

	return count

def currentDate():
	now = datetime.datetime.now()
	nowArr = str(now).split(" ") # Date, time
	dmy = nowArr[0].split("-") # Year, month, day
	y, m, d = map(int, dmy)

	return "{:0>2}".format(d) + "/" + "{:0>2}".format(m) + "/" + "{:0>4}".format(y)

def ordinal(n):
	if 10 <= n <= 20: return "th"

	lastDigit = n % 10

	if lastDigit == 1: return "st"
	elif lastDigit == 2: return "nd"
	elif lastDigit == 3: return "rd"
	else: return "th"

def readFile():
	file = open(CALENDAR_PATH, "r")
	rows = file.readlines()
	file.close()

	# Convert to grid, and skip header
	rows = [line.split(",") for line in rows[1:]]

	# Convert each line to an event
	return [Event(date, title, int(timeOld)) for date, title, timeOld in rows]

def writeFile(calendar):
	file = open(CALENDAR_PATH, "w")
	file.write("Date,Title,Days since 01/01/2000")

	for ev in calendar.events:
		file.write("\n{},{},{}".format(ev.date, ev.title, ev.timeOld))

	file.close()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

print("Reading calendar CSV file...\n")
calendar = Calendar(readFile())
print(len(calendar.events), "events\n")

calendar.sortEvents()

while True:
	choice = input("Enter: A to add an event,"
		+ "\n D to display calendar,"
		+ "\n R to remove events,"
		+ "\n or X to exit: ").upper()
	print()

	if choice == "A": # Add event
		date = "?"
		while not dateValid(date) and date != "X":
			date = input("Enter the date of the event (DD/MM/YYYY; must be in future) or X to exit: ").upper()

		if date != "X":
			d, m, y = map(int, date.split("/"))

			timeOld = daysSince112000(date)
			# E.g. "Tuesday 18th May 1999"
			dayDate = DAYS_OF_WEEK[(timeOld - 2) % 7] + " " + str(d) + ordinal(d) + " " + MONTHS[m - 1] + " " + str(y)

			title = input("Enter the event title for " + dayDate + ": ")

			calendar.events.append(Event(date, title, timeOld))
			calendar.sortEvents()

			writeFile(calendar)

			print("\nEvent added:\n" + str(newEvent))

	elif choice == "D": # Display calendar
		print(str(calendar))

	elif choice == "R": # Remove events
		if len(calendar.events) == 0:
			print("No events found")
		else:
			print(str(calendar) + "\n")

			choice = "?"
			while choice not in "1PA":
				choice = input("Enter 1 to delete 1 event, P for past, A for all: ").upper()
			print()

			if choice == "1":
				n = "?"
				validInput = False
				while not validInput:
					n = input("Enter the event no. that you wish to delete: ")
					if intValid(n):
						n = int(n)
						validInput = (1 <= n <= len(calendar.events))

				ev = calendar.events[n - 1]
				yn = "?"
				while yn not in "YN":
					yn = input("\nAre you sure you want to remove event:\n" + str(ev) + "? (Y/N) ").upper()

				if yn == "Y":
					calendar.events.remove(ev)
					print("\nEvent deleted")
			elif choice in "PA":
				quant = "past" if choice == "P" else "all"

				yn = "?"
				while yn not in "YN":
					yn = input("Are you sure you want to remove {} events? (Y/N) ".format(quant)).upper()

				if yn == "Y":
					if choice == "P": calendar.removePastEvents()
					else: calendar.events.clear()

					print("\n" + quant.capitalize() + " events removed")

			writeFile(calendar)

	elif choice == "X":
		break

	print()
