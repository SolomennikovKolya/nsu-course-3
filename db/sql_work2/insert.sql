
USE solomennikov22204work3;

INSERT INTO Employee (LastName, FirstName, MiddleName, Address) VALUES
('Ivanov', 'Sergej', 'Petrovich', 'Morskoj prospekt 12'),
('Ustinov', 'Oleg', 'Viktorovich', 'Krasnyj prospekt 20'),
('Kim', 'Dmitrij', 'Konstantinovich', 'Ul Pirogova 20'),
('Markova', 'Ekaterina', 'Andreevna', 'Ul Furmanova 12'),
('Sidorov', 'Aleksej', 'Andreevich', 'Ul Geroev truda'),
('Akopjan', 'Vasilij', 'Ilich', 'Ul Gogolja');

INSERT INTO Post (Title, HourlyRate) VALUES
('Bujgalker', 25.00),
('Inzhener', 40.00),
('Programmist', 60.00),
('Povar', 20.00),
('Barmen', 15.00),
('Menedzher', 70.00);

INSERT INTO Workplace (OrganizationName, Address, PhoneNumber, PensionContributions) VALUES
('Artehk', 'Novosibirsk Metro Krasnyj prospekt', '23-23-45', 10.00),
('Ajron', 'Moskva Taganskaja 9', '345-23-21', 15.00),
('Gotti', 'Novosibirsk  Ploschad Kondratjuka', '34-23-12', 12.00),
('Landor', 'Novosibirsk  Mikrorajon Sch', '43-78-90', 10.00);

INSERT INTO Job (EmployeeID, WorkplaceID, AssignmentDate, PostID, HoursWorked) VALUES
(1, 1, '2002-05-22', 2, 8),
(2, 1, '2002-05-26', 1, 6),
(3, 2, '2002-06-10', 3, 8),
(4, 3, '2002-05-10', 4, 6),
(5, 3, '2002-06-20', 5, 10),
(6, 4, '2002-07-30', 6, 8),
(1, 1, '2002-07-29', 6, 8),
(3, 2, '2002-12-17', 6, 8);