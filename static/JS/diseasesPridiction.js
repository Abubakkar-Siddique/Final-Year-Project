function onload(){
    var width = $(window).width();
    document.getElementById("head").css("width",width);
    document.getElementById("container-div").css("width",width);
    document.getElementById("homeLeft").css("width" , width/2);
    document.getElementById("homeRight").css("width" , width/2);
}

$(document).ready(function () {
  const inpFile = document.getElementById("inpFilee");
  const previewContainer = document.getElementById("imagePreview");
  const previewImage = previewContainer.querySelector(
    ".image-preview__image"
  );
  const previewDefaultText = previewContainer.querySelector(
    ".image-preview__default-text"
  );

  inpFile.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
      
      const reader = new FileReader();

      previewDefaultText.style.display = "none";
      previewImage.style.display = "block";
      reader.addEventListener("load", function () {
        previewImage.setAttribute("src", this.result);
      });

      reader.readAsDataURL(file);
    }
    
      
    
  });
});

function hover(id){
    document.getElementById(id).style.hover("border-bottom" , "10px");
}

/* Demo purposes only */
$(".hover").mouseleave(
    function () {
      $(this).removeClass("hover");
    }
  );


 function loadFile(event) {
	var image = document.getElementById('output');
	image.src = URL.createObjectURL(event.target.files[0]);
};  