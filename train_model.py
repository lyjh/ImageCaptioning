import caption_generator
from keras.callbacks import ModelCheckpoint, EarlyStopping

batch_size = 128

cg = caption_generator.CaptionGenerator()
cg.use_word_embedding = True
image_caption_model = cg.create_model()

# if weight != None:
#     model.load_weights(weight)

image_caption_model.load_weights('weights/weights-improvement-13-2.99.hdf5')
file_name = 'weights/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto')

train_generator = cg.data_generator(batch_size=batch_size, mode='train')
val_generator = cg.data_generator(batch_size=batch_size, mode='val')
image_caption_model.fit_generator(train_generator, steps_per_epoch=cg.total_samples//batch_size, validation_data=val_generator, validation_steps=cg.validation_samples//batch_size, epochs=30, verbose=1, callbacks=[checkpoint, early])