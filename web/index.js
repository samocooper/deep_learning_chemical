document.body.innerHTML =
  "Connecting to server..." 


$(function() {
  var socket = io('http://hackathon:3000');
  socket.on('id', function (event) {
    console.log(event.id);
    document.body.innerHTML = contentUpload();
    uploadListener($('#file'), socket);
  });

});

function uploadListener(input, socket) {
  input.change(function(e) {
    console.log('Uploading file');
    var file = e.target.files[0];
    var stream = ss.createStream();

    // upload a file to the server.
    ss(socket).emit('file', stream, {size: file.size});
    ss.createBlobReadStream(file).pipe(stream);
  });
}

function contentUpload() {
  return  '<input id="file" type="file" accept="*">'
}
