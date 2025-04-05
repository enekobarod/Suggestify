import tkinter as tk
from PIL import Image, ImageTk

# Función que manejará los gestos
def handle_gesture(gesture):
    if gesture == 'right':
        print("¡Me ha gustado! (1)")
    elif gesture == 'left':
        print("No me ha gustado (-1)")
    elif gesture == 'up':
        print("Saltar")

    #imagen, titulo, autor= llamado a recomendador
    #return imagen, titulo, autor


class GestureApp:
    def __init__(self, root, imagen, titulo, autor):
        self.root = root
        self.root.title("Interfaz de Gestos")

        # Cargar imagen
        self.image = Image.open(imagen)  # Cambia el nombre del archivo si es necesario
        self.image = self.image.resize((400, 600), Image.Resampling.LANCZOS)
        self.img_display = ImageTk.PhotoImage(self.image)

        # Canvas para manejar los gestos
        self.canvas = tk.Canvas(self.root, width=400, height=600)
        self.canvas.create_image(200, 300, image=self.img_display)
        self.canvas.pack()

        # Título y autor
        self.title_label = tk.Label(self.root, text=titulo, font=("Arial", 24))
        self.title_label.pack()
        self.author_label = tk.Label(self.root, text=autor, font=("Arial", 18))
        self.author_label.pack()

        # Añadir eventos para detectar gestos
        self.canvas.bind("<B1-Motion>", self.on_drag)  # Para detectar movimiento del mouse
        self.canvas.bind("<ButtonRelease-1>", self.on_release)  # Detectar cuando se suelta el clic
        self.x_start = None  # Guardamos la posición inicial
        self.gesture = None  # Para guardar el gesto detectado

    def on_drag(self, event):
        if self.x_start is None:
            self.x_start = event.x

    def on_release(self, event):
        if self.x_start:
            if event.x - self.x_start > 50:  # Si el movimiento es hacia la derecha
                self.gesture = 'right'
            elif event.x - self.x_start < -50:  # Si el movimiento es hacia la izquierda
                self.gesture = 'left'
            elif event.y - self.x_start > 50:  # Movimiento hacia arriba (saltar)
                self.gesture = 'up'
            self.handle_gesture(self.gesture)
            self.x_start = None  # Resetear

    def handle_gesture(self, gesture):
        handle_gesture(gesture)

# Crear la ventana principal
root = tk.Tk()
app = GestureApp(root, "caratula.jpg", "mi cancion", "nora")
root.mainloop()
