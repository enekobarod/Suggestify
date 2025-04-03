import tkinter as tk
from PIL import Image, ImageTk
from model import RecommenderModel

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interfaz de Gestos")

        # Instantiate the model
        #   We'll say we want 20 ratings before final recommendations:
        self.model = RecommenderModel(min_ratings_needed=20)

        # Variables to store current track info
        self.current_track = None

        # Canvas for image (400x600)
        self.canvas = tk.Canvas(self.root, width=400, height=600)
        self.canvas.pack()

        # Title and author
        self.title_label = tk.Label(self.root, text="", font=("Arial", 24))
        self.title_label.pack()
        self.author_label = tk.Label(self.root, text="", font=("Arial", 18))
        self.author_label.pack()

        # Gesture detection
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.x_start = None

        # Show the first random track
        self.show_new_track()

    def show_new_track(self):
        """
        Get a random track from the model and display it in the GUI
        (unless we already have enough ratings).
        """
        if self.model.has_enough_ratings():
            # If we have enough, go straight to showing recommendations
            self.show_recommendations()
            return

        track_info = self.model.get_random_song()
        self.current_track = track_info  # remember which track is on screen

        # Load and display the image
        #image_path = track_info["image_path"]
        image_path = "/home/enekobarba/bigdata/Suggestify/Suggestify/Suggestify/Ficheros_py/caratula.jpg"
        self.image = Image.open(image_path)
        self.image = self.image.resize((400, 600), Image.Resampling.LANCZOS)
        self.img_display = ImageTk.PhotoImage(self.image)

        self.canvas.delete("all")  # clear old image
        self.canvas.create_image(200, 300, image=self.img_display)

        # Update labels
        self.title_label.config(text=track_info["track_name"])
        self.author_label.config(text=track_info["artist_name"])

    def on_drag(self, event):
        """Track the initial x-position where user clicks+drags."""
        if self.x_start is None:
            self.x_start = event.x

    def on_release(self, event):
        """
        On mouse release, compute the horizontal displacement
        to see if user swiped left or right, or check vertical if you like.
        """
        if self.x_start is not None:
            dx = event.x - self.x_start
            # We can also do vertical checks with event.y if we want an "up" gesture

            gesture = None
            # Decide threshold for left vs right
            if dx > 50:
                gesture = 'right'
            elif dx < -50:
                gesture = 'left'
            # Example for "up" gesture if you want:
            #   if (some vertical threshold) then gesture = 'up'

            if gesture:
                self.handle_gesture(gesture)

            self.x_start = None  # reset

    def handle_gesture(self, gesture):
        """
        Once we detect a gesture, we pass it to the model as a rating, then decide
        whether we show the next track or show final recommendations.
        """
        # 1) Submit rating
        if self.current_track is None:
            return  # safety check

        track_id = self.current_track["track_id"]
        self.model.submit_gesture_rating(track_id, gesture)

        # 2) Check if we have enough ratings
        if self.model.has_enough_ratings():
            self.show_recommendations()
        else:
            # Display next random track
            self.show_new_track()

    def show_recommendations(self):
        """
        Get final recommendations from the model, show them in the GUI (or console).
        """
        recs = self.model.get_final_recommendations(top_n=5, diversity=0.3)
        print("\n*** Recomendaciones finales ***")
        print(recs[["track_name", "artist_name"]])

        # For demonstration, let's just display the first recommended track in the GUI
        if len(recs) > 0:
            first_rec = recs.iloc[0]
            self.title_label.config(text="RECOMENDADO: " + first_rec["track_name"])
            self.author_label.config(text=first_rec["artist_name"])

            # If you have an image for it, put the path here. For now, re-use "caratula.jpg".
            self.image = Image.open("/home/enekobarba/bigdata/Suggestify/Suggestify/Suggestify/Ficheros_py/caratula.jpg")
            self.image = self.image.resize((400, 600), Image.Resampling.LANCZOS)
            self.img_display = ImageTk.PhotoImage(self.image)

            self.canvas.delete("all")
            self.canvas.create_image(200, 300, image=self.img_display)
        else:
            self.title_label.config(text="No hay recomendaciones")
            self.author_label.config(text="")

# Entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
