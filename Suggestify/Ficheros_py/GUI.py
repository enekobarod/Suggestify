import tkinter as tk
from PIL import Image, ImageTk
from model import RecommenderModel
import os

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interfaz de Gestos")
        
        #tamaño fijo
        self.root.geometry("400x700")
        self.root.resizable(False, False)
        
        self.model = RecommenderModel(min_ratings_needed=20)
        self.current_track = None
        self.drag_data = {"x": 0, "y": 0, "item": None}
        self.animation_in_progress = False
        self.title_animation_id = None
        self.title_position = 0
        
        #paleta de colores
        self.COLOR_PRINCIPAL = "#2E3440"
        self.COLOR_SECUNDARIO = "#434C5E"
        self.COLOR_TEXTO = "#ECEFF4"
        self.COLOR_ACENTO = "#88C0D0"
        self.COLOR_FEEDBACK = "#BF616A"

        root.configure(bg=self.COLOR_PRINCIPAL)
        
        #hueco pa la imagen
        self.canvas = tk.Canvas(
            root, 
            width=400, 
            height=600, 
            bg=self.COLOR_PRINCIPAL,
            highlightthickness=0
        )
        self.canvas.pack()

        text_frame = tk.Frame(self.root, height=100, width=400)
        text_frame.pack_propagate(False)
        text_frame.pack()
        
        # Etiquetas con desplazamiento automático
        self.title_label = tk.Label(text_frame, text="", font=("Arial", 20), wraplength=380, justify="center")
        self.title_label.pack(pady=(10, 0))
        self.author_label = tk.Label(text_frame, text="", font=("Arial", 16))
        self.author_label.pack()

        #events
        self.canvas.bind("<Button-1>", self.on_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.show_new_track()
        self.start_title_scroll()


    def show_new_track(self):
        #mostrar nueva canción
        if self.model.has_enough_ratings():
            self.show_recommendations()
            return

        track_info = self.model.get_random_song()
        self.current_track = track_info

        #ponerle caratula
        self.image = Image.open("caratula.jpg").resize((400, 600), Image.Resampling.LANCZOS)
        self.img_display = ImageTk.PhotoImage(self.image)

        self.canvas.delete("all")
        self.track_image = self.canvas.create_image(200, 300, image=self.img_display)

        #update de titulo y autor
        self.title_label.config(text=track_info["track_name"])
        self.author_label.config(text=track_info["artist_name"])
        self.start_title_scroll()



#movidas de las animaciones
    def start_title_scroll(self):
        #teleprompter en el titulo
        if self.title_animation_id:
            self.root.after_cancel(self.title_animation_id)
        
        self.title_position = 0
        self.animate_title_scroll()

    def animate_title_scroll(self):
        #animacion teleprompter
        display_length = 20
        if len(self.title_text) > display_length:
            self.title_position += 1
            if self.title_position > len(self.title_text):
                self.title_position = 0  #reinicia

            visible_text = self.title_text[self.title_position:] + self.title_text[:self.title_position]
            self.title_label.config(text=visible_text[:display_length])

        self.root.after(100, self.animate_title_scroll)

    def on_start(self, event):
        if self.animation_in_progress:
            return
            
        self.drag_data = {"x": event.x, "y": event.y, "item": self.track_image}

    def on_drag(self, event):
        if self.animation_in_progress or not self.drag_data["item"]:
            return
            
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        
        self.canvas.move(self.drag_data["item"], dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def show_feedback_animation(self, direction):
        #animacion de gesto
        self.canvas.delete("feedback")
        
        img_map = {
            'right': 'heart.png',
            'left': 'x_mark.png',
            'up': 'jump_icon.png'
        }
        
        try:
            img = Image.open(img_map[direction])
            TARGET_SIZE = (100, 100)
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            feedback_img = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return
        
        self.feedback_img = feedback_img 
        self.canvas.create_image(
            200, 300,
            image=self.feedback_img,
            tags="feedback",
            anchor="center"
        )
        self.root.after(500, lambda: self.canvas.delete("feedback"))

    def on_release(self, event):
        #fin de gesto
        if self.animation_in_progress:
            return
            
        pos = self.canvas.coords(self.track_image)
        if not pos:
            return
            
        x, y = pos
        gesture = None
        
        #determinar gesto
        if x > 400:
            gesture = 'right'
        elif x < 0:
            gesture = 'left'
        elif y < 150:
            gesture = 'up'
            
        if gesture:
            self.show_feedback_animation(gesture)
            self.root.after(500, lambda: self.handle_gesture(gesture))
        else:
            self.animate_return_to_center()

    def animate_return_to_center(self):
        #suavidad en movimientos
        self.animation_in_progress = True
        
        def update_animation():
            pos = self.canvas.coords(self.track_image)
            dx = 200 - pos[0]
            dy = 300 - pos[1]
            
            if abs(dx) > 1 or abs(dy) > 1:
                self.canvas.move(self.track_image, dx*0.2, dy*0.2)
                self.root.after(16, update_animation)
            else:
                self.canvas.coords(self.track_image, 200, 300)
                self.animation_in_progress = False
        
        update_animation()

    def handle_gesture(self, gesture):
        #gesto
        if self.current_track is None:
            return
            
        self.model.submit_gesture_rating(self.current_track["track_id"], gesture)
        
        if self.model.has_enough_ratings():
            self.show_recommendations()
        else:
            self.show_new_track()

    def show_recommendations(self):
        #mostrar recomendaciones
        recs = self.model.get_final_recommendations(top_n=5, diversity=0.3)
        print("\n*** Recomendaciones finales ***")
        print(recs[["track_name", "artist_name"]])

        if len(recs) > 0:
            first_rec = recs.iloc[0]
            self.title_label.config(text="RECOMENDADO: " + first_rec["track_name"])
            self.author_label.config(text=first_rec["artist_name"])

            self.image = Image.open("caratula.jpg")
            self.image = self.image.resize((400, 600), Image.Resampling.LANCZOS)
            self.img_display = ImageTk.PhotoImage(self.image)

            self.canvas.delete("all")
            self.canvas.create_image(200, 300, image=self.img_display)
        else:
            self.title_label.config(text="No hay recomendaciones")
            self.author_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()