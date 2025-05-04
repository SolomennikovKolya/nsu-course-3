import React, { useEffect, useState } from 'react';
import { useLocation, useParams } from 'react-router-dom';
import axios from '../../axios';

function Equipment() {
    const { categoryNameSlug, equipmentNameSlug } = useParams();     // Slug названия из URL
    const [categoryName, setCategoryName] = useState(null);          // Название категории
    const [equipmentName, setEquipmentName] = useState(null);        // Название оборудования
    const [equipmentData, setEquipmentData] = useState(null);        // Данные оборудования
    const [loading, setLoading] = useState(true);                    // Идёт ли загрузка
    const [error, setError] = useState(null);                        // Ошибка
    const [isModalOpen, setIsModalOpen] = useState(false);           // Состояние для отображения модального окна
    const [isButtonDisabled, setIsButtonDisabled] = useState(false); // Для деактивации кнопки
    const [bookingError, setBookingError] = useState(null);          // Ошибка бронирования
    const [formData, setFormData] = useState({ name: '', phone: '', email: '', rentFrom: '', rentTo: '' }); // Состояние для данных формы

    useEffect(() => {
        if (!equipmentName) return;
        const fetchEquipment = async () => {
            try {
                const res = await axios.get('/equipment', {
                    params: { name: equipmentName }
                });
                setEquipmentData(res.data);
                setIsButtonDisabled(res.data.available_count == 0); // Деактивируем кнопку, если оборудования нет в наличии
            } catch (error) {
                console.error('Ошибка загрузки информации об оборудовании:', error);
                setError('Не удалось загрузить информацию об оборудовании.');
            } finally {
                setLoading(false);
            }
        };
        fetchEquipment();
    }, [equipmentName]);

    // Получаем переданные данные через state (название категории и оборудования)
    const location = useLocation();
    useEffect(() => {
        if (location.state?.categoryName) {
            setCategoryName(location.state.categoryName);
        }
        if (location.state?.equipmentName) {
            setEquipmentName(location.state.equipmentName);
        }
    }, [location.state]);

    // Обработчик отправки формы
    const handleFormSubmit = async (e) => {
        e.preventDefault();
        try {
            const res = await axios.post('/book_equipment', {
                equipmentName: equipmentName,
                ...formData
            });
            setIsModalOpen(false);
            setBookingError(null);

            if (equipmentData.available_count == 1) {
                setIsButtonDisabled(true);
            }
            setEquipmentData(prevData => ({
                ...prevData,
                available_count: prevData.available_count - 1
            }));
            alert("Оборудование успешно забронировано");
        } catch (error) {
            console.error('Ошибка бронирования:', error);
            setBookingError(error?.response?.data?.error);
        }
    };

    // Обработчик изменения данных в форме
    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData((prevData) => ({
            ...prevData,
            [name]: value
        }));
    };

    // Обработчик клика по фону модального окна (для его закрытия)
    const handleCloseModal = (e) => {
        if (e.target.classList.contains('modal')) {
            setIsModalOpen(false);
        }
    };

    return (
        <div style={{ padding: '0 2rem 2rem 2rem' }}>
            {categoryName !== null && equipmentName !== null && (<p className="subtext" style={{ margin: '0.5rem 0 0.5rem 0' }}> Каталог → {categoryName} → {equipmentName} </p>)}

            {loading ? (
                <p></p>
            ) : error ? (
                <p className="error-message">{error}</p>
            ) : (
                <div style={{ display: 'flex' }}>
                    {/* Описание оборудования */}
                    <div style={{ flexGrow: 1, flexBasis: 0, padding: '0 1rem 0 0' }}>
                        <h1 className="page-title">{equipmentName}</h1>
                        <p>{equipmentData.description}</p>
                    </div>

                    {/* Бронирование */}
                    <div style={{ width: '300px', padding: '1rem' }}>
                        <p>Цена аренды: {equipmentData.rental_price_per_day} ₽/день</p>
                        <p>Залог: {equipmentData.deposit_amount} ₽</p>
                        <button
                            className={isButtonDisabled ? "action-button-disabled" : "action-button"}
                            onClick={() => setIsModalOpen(true)}
                            disabled={isButtonDisabled}
                        >Забронировать
                        </button>
                        <p style={{ color: 'grey' }}>{equipmentData.available_count} шт. в наличии</p>
                    </div>
                </div>
            )}

            {/* Модальное окно для бронирования */}
            {isModalOpen && (
                <div className="modal" onMouseDown={handleCloseModal}>
                    <div className="modal-content">
                        <h2>{equipmentName}</h2>
                        {bookingError !== null && (
                            <p className='error-message'>{bookingError}</p>
                        )}
                        <form onSubmit={handleFormSubmit}>
                            <div>
                                <label>Личные данные:</label>
                                <input className='text-input' placeholder='Имя' type="text" id="name" name="name" value={formData.name} onChange={handleInputChange} required />
                                <input className='text-input' placeholder='Телефон' type="text" id="phone" name="phone" value={formData.phone} onChange={handleInputChange} required />
                                <input className='text-input' placeholder='Email' type="email" id="email" name="email" value={formData.email} onChange={handleInputChange} required />
                            </div>
                            <div>
                                <label>Даты аренды (от и до включительно):</label>
                                <input className='text-input' type="date" id="rentFrom" name="rentFrom" value={formData.rentFrom} onChange={handleInputChange} required />
                                <input className='text-input' type="date" id="rentTo" name="rentTo" value={formData.rentTo} onChange={handleInputChange} required />
                            </div>
                            <button type="submit" className='action-button'>Забронировать</button>
                            <a className='link-style' href="/terms" target="_blank" rel="noopener noreferrer">Условия аренды</a>
                        </form>
                    </div>
                </div>
            )}

        </div>
    );
}

export default Equipment;
