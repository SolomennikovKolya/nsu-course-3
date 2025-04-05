
USE solomennikov22204work3;

-- Запрос 1
SELECT 
    e.LastName, e.FirstName, e.MiddleName, 
    p.Title AS Position, 
    w.OrganizationName AS Workplace, 
    MAX(j.AssignmentDate) AS LastAssignmentDate
FROM Employee e
JOIN Job j ON e.EmployeeID = j.EmployeeID
JOIN Post p ON j.PostID = p.PostID
JOIN Workplace w ON j.WorkplaceID = w.WorkplaceID
GROUP BY e.EmployeeID, p.Title, w.OrganizationName
ORDER BY w.OrganizationName;

-- Запрос 2
UPDATE Post
SET HourlyRate = HourlyRate * 2
WHERE Title IN ('Programmist', 'Menedzher');
SELECT 
    e.LastName, e.FirstName, e.MiddleName, 
    p.Title, p.HourlyRate
FROM Employee e
JOIN Job j ON e.EmployeeID = j.EmployeeID
JOIN Post p ON j.PostID = p.PostID
WHERE p.Title IN ('Programmist', 'Menedzher');

-- Запрос 3
SELECT 
    w.OrganizationName, 
    COUNT(DISTINCT j.EmployeeID) AS EmployeeCount
FROM Workplace w
JOIN Job j ON w.WorkplaceID = j.WorkplaceID
GROUP BY w.OrganizationName
ORDER BY w.OrganizationName;

-- Запрос 4
SELECT 
    e.LastName, e.FirstName, e.MiddleName, 
    p.Title AS Position, 
    w.OrganizationName AS Workplace, 
    (p.HourlyRate * j.HoursWorked * 21) AS MonthlySalary
FROM Employee e
JOIN Job j ON e.EmployeeID = j.EmployeeID
JOIN Post p ON j.PostID = p.PostID
JOIN Workplace w ON j.WorkplaceID = w.WorkplaceID
WHERE j.AssignmentDate < '2002-06-12';

-- Запрос 5
SELECT 
    e.FirstName, e.LastName, 
    (p.HourlyRate * j.HoursWorked * 21 * (w.PensionContributions / 100)) AS PensionContribution
FROM Employee e
JOIN Job j ON e.EmployeeID = j.EmployeeID
JOIN Post p ON j.PostID = p.PostID
JOIN Workplace w ON j.WorkplaceID = w.WorkplaceID;
