package main

import (
	"image/color"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	_ "github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// Game implements ebiten.Game interface.
type Game struct{}

// Update proceeds the game state.
// Update is called every tick (1/60 [s] by default).
func (g *Game) Update() error {
	// Write your game's logical update.
	return nil
}

// Draw draws the game screen.
// Draw is called every frame (typically 1/60[s] for 60Hz display).
func (g *Game) Draw(screen *ebiten.Image) {
	// Write your game's rendering.
	// ebitenutil.DebugPrint(screen, "Hello, World!")

	// Ширина и высота прямоугольника
	// rectWidth, rectHeight := 620, 460
	rectWidth, rectHeight := 300, 220

	// Создаем изображение для прямоугольника
	rectImage := ebiten.NewImage(rectWidth, rectHeight)

	// Задаем цвет (серый)
	grayColor := color.RGBA{128, 128, 128, 255} // RGB 128, 128, 128 — серый
	rectImage.Fill(grayColor)

	// Параметры рисования
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(10, 10) // Координаты X и Y для рисования

	// Рисуем прямоугольник на экране
	screen.DrawImage(rectImage, op)
}

// Layout takes the outside size (e.g., the window size) and returns the (logical) screen size.
// If you don't have to adjust the screen size with the outside size, just return a fixed size.
func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 320, 240
	// return 640, 480
}

func main() {
	game := &Game{}
	// Specify the window size as you like. Here, a doubled size is specified.
	ebiten.SetWindowSize(640, 480)
	ebiten.SetWindowTitle("Your game's title")
	// Call ebiten.RunGame to start your game loop.
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
