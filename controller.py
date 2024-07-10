import traceback
import time
import asyncio
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from factories import FactoryReaders, FactorySearchEngines, FactorySaver
from reader import BaseReader
from search_engine import BaseSearchEngine
from saver import BaseSaver
import app_logger 


logger = app_logger.get_logger(__name__)

class Controller:
    def __init__(self, config, factory_readers: FactoryReaders, factory_savers: FactorySaver, factory_search_engines: FactorySaver) -> None:
        self._count_frames = 0
        self._queue_read = Queue(1000)
        self._queue_save = Queue(1000)
        self._config = config
        self._batch = config['controller']['batch']
        self._reader = self._init_reader(factory_readers)
        self._search_engine = self._init_search_engine(factory_search_engines)
        self._saver = self._init_saver(factory_savers)
        self._stop_reading = False
        self._stop_saving = False
        self._main_loop_end = False
        self._need_save = self._config['saver']['init']['need_save']
        self._count_save_frames = 0

    def _init_reader(self, factory_readers: FactoryReaders) -> BaseReader:
        return factory_readers(**self._config['reader']['init'])

    def _init_search_engine(self, factory_search_engines: FactorySearchEngines) -> BaseSearchEngine:
        return factory_search_engines(**self._config['search_engine']['init'])

    def _init_saver(self, factory_savers: FactorySaver) -> BaseSaver:
        return factory_savers(**self._config['saver']['init'])
    
    def _is_end_main_loop(self) -> bool:
        return self._queue_read.empty() and self._stop_reading

    def run(self) -> None:
        start_time = time.time() 
        with ThreadPoolExecutor(max_workers=3) as e:
            # asyncio.run(self._read_frame())
            # for _ in range(self._num_thread_for_read):
            e.submit(asyncio.run, self._read_frames())
            # asyncio.run(self._main_loop())
            e.submit(asyncio.run, self._main_loop())
            # asyncio.run(self._save_images())
            # for _ in range(self._num_thread_for_save):
            e.submit(asyncio.run, self._save_images())
        end_time = time.time() 
        logger.info(f'All time {end_time - start_time}')
        logger.info(f'Count frames {self._count_frames}')
        logger.info(f'Count save frames {self._count_save_frames}')
        mean_time_frame = (end_time - start_time) / self._count_frames
        logger.info(f'FPS {1. / mean_time_frame}')
        logger.info(f'Mean time for frame {mean_time_frame}')

    async def _main_loop(self):
        try:
            self._main_loop_end = False
            while True:
                if self._is_end_main_loop():
                    break
                data = await self._collect_batch(self._batch)
                logger.info(f'main loop collect batch. Queue read size: {self._queue_read.qsize()}')
                if self._batch == 1 and len(data) == 1:
                    unique_images = await self._search_engine.run(data[0])
                    if unique_images is None:
                        unique_images = []
                elif self._batch > 1:    
                    unique_images = await self._search_engine.run_batch(data)
                else:
                    raise ValueError(f'Batch less 1 or len(data): f{len(data)} not eq batch_size: f{self._batch}')
                logger.info('main loop search unique images')
                if self._need_save:
                    for img in unique_images:
                        self._queue_save.put(img)
                        # await self._save_images(unique_images)
                logger.info(f'main loop queue save put. Queue save size: {self._queue_save.qsize()}')
            self._main_loop_end = True
            logger.info('main loop end')
        except Exception as e:
            logger.error('main loop', exc_info=True)
        self._queue_save.put(None)
                
    async def _read_frames(self):
        try:
            self._stop_reading = False
            kwargs_read_data = self._config['reader'].get('read_data', None)
            if kwargs_read_data is None:
                kwargs_read_data = dict()
            for img_with_name in self._reader.read_data(**kwargs_read_data):
                self._queue_read.put(img_with_name)
                self._count_frames += 1
                logger.debug('read frames queue read put')
            logger.info('read frames stop reading True')
            self._stop_reading = True
        except Exception as e:
            logger.error(f'read frames', exc_info=True)

    async def _save_images(self):
        self._stop_saving = False
        try:
            while True:
                if self._stop_reading and self._queue_save.empty() and self._main_loop_end:
                    break
                if self._stop_reading and self._queue_save.empty():
                    logger.debug('save images continue')
                    continue
                img_with_name = self._queue_save.get()
                if img_with_name is None:
                    break
                logger.debug('save images queue save get')
                await self._saver.save(img_with_name)
                self._count_save_frames += 1
                logger.debug(f'save images save img: {img_with_name.name}')
            logger.info('save images end')
        except Exception as e:
            logger.error(f'save images', exc_info=True)
        self._stop_saving = True
   
    async def _collect_batch(self, batch_size):
        data = []
        for _ in range(batch_size):
            if self._is_end_main_loop():
                return data
            img_with_name = self._queue_read.get()
            data.append(img_with_name)
        return data
