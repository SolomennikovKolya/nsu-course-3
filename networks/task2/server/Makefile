all: exec

main.exe: *.go
	go build -o main.exe

exec: main.exe
	main.exe -addr="127.0.0.1" -port="8080"

run:
	go run main.go -addr="127.0.0.1" -port="8080"

clean:
	del main.exe