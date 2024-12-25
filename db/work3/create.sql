
CREATE DATABASE solomennikov22204work3;
USE solomennikov22204work3;

-- Сотрудники
CREATE TABLE Employee (
    EmployeeID INT AUTO_INCREMENT PRIMARY KEY,
    LastName VARCHAR(50) NOT NULL,
    FirstName VARCHAR(50) NOT NULL,
    MiddleName VARCHAR(50),
    Address TEXT NOT NULL
);

-- Должности
CREATE TABLE Post (
    PostID INT AUTO_INCREMENT PRIMARY KEY,
    Title VARCHAR(100) NOT NULL,
    HourlyRate DECIMAL(10, 2) NOT NULL CHECK (HourlyRate > 0)
);

-- Места работы
CREATE TABLE Workplace (
    WorkplaceID INT AUTO_INCREMENT PRIMARY KEY,
    OrganizationName VARCHAR(100) NOT NULL,
    Address TEXT NOT NULL,
    PhoneNumber VARCHAR(20),
    PensionContributions DECIMAL(5, 2) NOT NULL CHECK (PensionContributions >= 0 AND PensionContributions <= 100)
);

-- Работы
CREATE TABLE Job (
    JobID INT AUTO_INCREMENT PRIMARY KEY,
    EmployeeID INT NOT NULL,
    WorkplaceID INT NOT NULL,
    AssignmentDate DATE NOT NULL,
    PostID INT NOT NULL,
    HoursWorked INT NOT NULL CHECK (HoursWorked >= 0),
    FOREIGN KEY (EmployeeID) REFERENCES Employee(EmployeeID) ON DELETE CASCADE,
    FOREIGN KEY (WorkplaceID) REFERENCES Workplace(WorkplaceID) ON DELETE CASCADE,
    FOREIGN KEY (PostID) REFERENCES Post(PostID) ON DELETE CASCADE
);

